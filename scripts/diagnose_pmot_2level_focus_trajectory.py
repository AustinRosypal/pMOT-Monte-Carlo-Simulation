"""Diagnose focused-beam artifacts in two effective two-level pMOT launches.

This is a provisional-model diagnostic, not a physical pMOT prediction.  It
compares the on-axis launch through an ideal 2.2-micrometre trapping waist with
an otherwise identical trajectory displaced by 0.5 mm in y.  It records the
trajectory, local force derivatives, Stark terms, beam-resolved scattering,
static axial force landscapes, and a short timestep-refinement audit.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pmot.configuration import PLANCK_CONSTANT_J_S
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pMOT_2Level import simulate_pmot_2level_trajectory
from pmot.state import AtomState


OUTPUT_ROOT = Path(
    "outputs/diagnostics/pMOT_2Level/focus_trajectory_diagnosis_20261016"
)
PATH_POWERS_W = np.asarray((19.147243e-3, 19.147243e-3, 38.294486e-3))
COOLING_POWER_W = 27.0e-3
COOLING_DETUNING_HZ = -15.0e6
DURATION_S = 25.0e-3
TIME_STEP_S = 2.5e-6
DIAGNOSTIC_STRIDE = 10
POSITION_DIFFERENCE_STEP_M = 2.0e-6
VELOCITY_DIFFERENCE_STEP_M_PER_S = 0.01


def _build_model():
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=COOLING_DETUNING_HZ,
        include_gravity=True,
    )
    baseline = default_simple_mot_apparatus()
    apparatus = replace(
        baseline,
        cooling=replace(
            baseline.cooling,
            power_w_per_beam=COOLING_POWER_W,
            detuning_hz=COOLING_DETUNING_HZ,
            beam_diameter_m=12.7e-3,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    return simple, cooling, trapping, table


def _record_arrays(record):
    return {
        "time_s": np.asarray(record.times_s),
        "position_m": np.asarray(record.positions_m),
        "velocity_m_per_s": np.asarray(record.velocities_m_per_s),
        "force_n": np.asarray(record.forces_n),
        "quantization_axis": np.asarray(record.quantization_axes),
        "axis_defined": np.asarray(record.axis_defined_by_vector_shift),
        "sigma_plus_shift_hz": np.asarray(
            record.total_ac_stark_shifts_hz_for_sigma_plus
        ),
        "total_scattering_rate_per_s": np.asarray(
            record.total_scattering_rates_per_s
        ),
    }


def _evaluate(simple, cooling, trapping, table, position, velocity, previous_axis):
    return evaluate_pmot_2level_local_observable(
        cooling,
        trapping,
        position,
        velocity,
        path_powers_w=PATH_POWERS_W,
        previous_axis=previous_axis,
        simple_config=simple,
        polarizability_table=table,
    )


def _trajectory_diagnostics(simple, cooling, trapping, table, arrays):
    indices = np.arange(0, len(arrays["time_s"]), DIAGNOSTIC_STRIDE)
    if indices[-1] != len(arrays["time_s"]) - 1:
        indices = np.append(indices, len(arrays["time_s"]) - 1)
    n = len(indices)
    diagnostics = {
        "index": indices,
        "time_s": arrays["time_s"][indices],
        "position_m": arrays["position_m"][indices],
        "velocity_m_per_s": arrays["velocity_m_per_s"][indices],
        "force_n": arrays["force_n"][indices],
        "total_intensity_w_per_m2": np.empty(n),
        "scalar_shift_hz": np.empty(n),
        "vector_shift_hz": np.empty(n),
        "tensor_shift_hz": np.empty(n),
        "transition_shifts_hz": np.empty((n, 3)),
        "quantization_axis": np.empty((n, 3)),
        "position_force_gradient_x_n_per_m": np.empty(n),
        "velocity_force_gradient_x_n_s_per_m": np.empty(n),
        "x_beam_rates_per_s": np.empty((n, 2)),
        "x_beam_closest_detuning_hz": np.empty((n, 2)),
        "x_beam_doppler_hz": np.empty((n, 2)),
    }
    x_beam_indices = [
        index
        for index, beam in enumerate(cooling)
        if abs(float(beam.direction[0])) > 0.5
    ]
    x_beam_indices.sort(key=lambda index: cooling[index].direction[0], reverse=True)
    for out_index, record_index in enumerate(indices):
        position = arrays["position_m"][record_index]
        velocity = arrays["velocity_m_per_s"][record_index]
        previous_axis = arrays["quantization_axis"][record_index]
        observable = _evaluate(
            simple, cooling, trapping, table, position, velocity, previous_axis
        )
        stark = observable.stark
        diagnostics["total_intensity_w_per_m2"][out_index] = np.sum(
            stark.component_intensities_w_per_m2
        )
        diagnostics["scalar_shift_hz"][out_index] = (
            stark.total_scalar_energy_j / PLANCK_CONSTANT_J_S
        )
        diagnostics["vector_shift_hz"][out_index] = (
            stark.total_vector_energy_j / PLANCK_CONSTANT_J_S
        )
        diagnostics["tensor_shift_hz"][out_index] = (
            stark.total_tensor_energy_j / PLANCK_CONSTANT_J_S
        )
        diagnostics["transition_shifts_hz"][out_index] = (
            stark.transition_shifts_hz_by_q
        )
        diagnostics["quantization_axis"][out_index] = stark.quantization_axis

        displacement = np.asarray((POSITION_DIFFERENCE_STEP_M, 0.0, 0.0))
        plus = _evaluate(
            simple,
            cooling,
            trapping,
            table,
            position + displacement,
            velocity,
            stark.quantization_axis,
        )
        minus = _evaluate(
            simple,
            cooling,
            trapping,
            table,
            position - displacement,
            velocity,
            stark.quantization_axis,
        )
        diagnostics["position_force_gradient_x_n_per_m"][out_index] = (
            plus.force_n[0] - minus.force_n[0]
        ) / (2.0 * POSITION_DIFFERENCE_STEP_M)

        velocity_delta = np.asarray(
            (VELOCITY_DIFFERENCE_STEP_M_PER_S, 0.0, 0.0)
        )
        plus = _evaluate(
            simple,
            cooling,
            trapping,
            table,
            position,
            velocity + velocity_delta,
            stark.quantization_axis,
        )
        minus = _evaluate(
            simple,
            cooling,
            trapping,
            table,
            position,
            velocity - velocity_delta,
            stark.quantization_axis,
        )
        diagnostics["velocity_force_gradient_x_n_s_per_m"][out_index] = (
            plus.force_n[0] - minus.force_n[0]
        ) / (2.0 * VELOCITY_DIFFERENCE_STEP_M_PER_S)

        for direction_index, beam_index in enumerate(x_beam_indices):
            sample = observable.rate_samples[beam_index]
            weights = np.asarray(sample.polarization_weights_q_minus_pi_q_plus)
            detunings = np.asarray(
                sample.effective_detunings_hz_q_minus_pi_q_plus
            )
            allowed = weights > 1.0e-10
            diagnostics["x_beam_rates_per_s"][out_index, direction_index] = (
                sample.total_scattering_rate_per_s
            )
            diagnostics["x_beam_closest_detuning_hz"][
                out_index, direction_index
            ] = detunings[allowed][np.argmin(np.abs(detunings[allowed]))]
            diagnostics["x_beam_doppler_hz"][out_index, direction_index] = (
                sample.doppler_shift_hz
            )
    return diagnostics


def _axial_landscape(simple, cooling, trapping, table):
    x_m = np.linspace(-25.0e-3, 25.0e-3, 1201)
    offsets_m = (0.0, 0.5e-3)
    velocities_m_per_s = (0.0, -10.0)
    result = {"x_m": x_m}
    for y_m in offsets_m:
        for vx in velocities_m_per_s:
            key = f"y_{1e3*y_m:g}mm_vx_{vx:g}mps"
            force_x = np.empty_like(x_m)
            total_intensity = np.empty_like(x_m)
            transition_shifts = np.empty((len(x_m), 3))
            axis = np.empty((len(x_m), 3))
            previous_axis = np.asarray((1.0, 0.0, 0.0))
            for index, x_value in enumerate(x_m):
                observable = _evaluate(
                    simple,
                    cooling,
                    trapping,
                    table,
                    (x_value, y_m, 0.0),
                    (vx, 0.0, 0.0),
                    previous_axis,
                )
                previous_axis = np.asarray(observable.stark.quantization_axis)
                force_x[index] = observable.force_n[0]
                total_intensity[index] = np.sum(
                    observable.stark.component_intensities_w_per_m2
                )
                transition_shifts[index] = observable.stark.transition_shifts_hz_by_q
                axis[index] = observable.stark.quantization_axis
            result[f"{key}_force_x_n"] = force_x
            result[f"{key}_total_intensity_w_per_m2"] = total_intensity
            result[f"{key}_transition_shifts_hz"] = transition_shifts
            result[f"{key}_quantization_axis"] = axis
            result[f"{key}_force_gradient_x_n_per_m"] = np.gradient(force_x, x_m)
    return result


def _crossing(array, value):
    delta = array - value
    indices = np.flatnonzero(delta[:-1] * delta[1:] <= 0.0)
    return int(indices[0]) if len(indices) else None


def _summary(name, arrays, diagnostics):
    speed = np.linalg.norm(arrays["velocity_m_per_s"], axis=1)
    force_magnitude = np.linalg.norm(arrays["force_n"], axis=1)
    peak_force_index = int(np.argmax(force_magnitude))
    peak_intensity_index = int(np.argmax(diagnostics["total_intensity_w_per_m2"]))
    zero_velocity_index = _crossing(arrays["velocity_m_per_s"][:, 0], 0.0)
    center_index = _crossing(arrays["position_m"][:, 0], 0.0)
    restoring_fraction = np.mean(
        diagnostics["position_force_gradient_x_n_per_m"] < 0.0
    )
    damping_fraction = np.mean(
        diagnostics["velocity_force_gradient_x_n_s_per_m"] < 0.0
    )
    return {
        "case": name,
        "final_position_mm": (1.0e3 * arrays["position_m"][-1]).tolist(),
        "final_velocity_m_per_s": arrays["velocity_m_per_s"][-1].tolist(),
        "minimum_x_mm": float(1.0e3 * np.min(arrays["position_m"][:, 0])),
        "maximum_speed_m_per_s": float(np.max(speed)),
        "peak_force_zN": float(1.0e21 * force_magnitude[peak_force_index]),
        "peak_force_time_ms": float(1.0e3 * arrays["time_s"][peak_force_index]),
        "peak_force_position_mm": (
            1.0e3 * arrays["position_m"][peak_force_index]
        ).tolist(),
        "peak_sampled_intensity_w_per_m2": float(
            diagnostics["total_intensity_w_per_m2"][peak_intensity_index]
        ),
        "peak_sampled_intensity_position_mm": (
            1.0e3 * diagnostics["position_m"][peak_intensity_index]
        ).tolist(),
        "sigma_plus_shift_range_mhz": (
            1.0e-6
            * np.asarray(
                (
                    np.min(arrays["sigma_plus_shift_hz"]),
                    np.max(arrays["sigma_plus_shift_hz"]),
                )
            )
        ).tolist(),
        "scattering_rate_range_per_s": [
            float(np.min(arrays["total_scattering_rate_per_s"])),
            float(np.max(arrays["total_scattering_rate_per_s"])),
        ],
        "fraction_sampled_locally_restoring_x": float(restoring_fraction),
        "fraction_sampled_locally_damping_x": float(damping_fraction),
        "x_velocity_zero_time_ms": (
            None
            if zero_velocity_index is None
            else float(1.0e3 * arrays["time_s"][zero_velocity_index])
        ),
        "x_velocity_zero_position_mm": (
            None
            if zero_velocity_index is None
            else (1.0e3 * arrays["position_m"][zero_velocity_index]).tolist()
        ),
        "first_x_center_crossing_time_ms": (
            None
            if center_index is None
            else float(1.0e3 * arrays["time_s"][center_index])
        ),
    }


def _short_convergence(simple, cooling, trapping, table, initial_state):
    result = {}
    duration_s = 5.0e-3
    for time_step_us in (2.5, 1.25, 0.625, 0.3125):
        record = simulate_pmot_2level_trajectory(
            cooling,
            trapping,
            initial_state,
            duration_s,
            time_step_us * 1.0e-6,
            path_powers_w=PATH_POWERS_W,
            simple_config=simple,
            polarizability_table=table,
        )
        arrays = _record_arrays(record)
        result[f"{time_step_us:g}_us"] = {
            "final_position_mm": (1.0e3 * arrays["position_m"][-1]).tolist(),
            "final_velocity_m_per_s": arrays["velocity_m_per_s"][-1].tolist(),
            "maximum_speed_m_per_s": float(
                np.max(np.linalg.norm(arrays["velocity_m_per_s"], axis=1))
            ),
            "peak_force_zN": float(
                1.0e21 * np.max(np.linalg.norm(arrays["force_n"], axis=1))
            ),
        }
    return result


def _plot_trajectory_overview(cases, diagnostics, output_path):
    figure, axes = plt.subplots(4, 2, figsize=(13.5, 12.5), sharex="col")
    colors = {"on_axis": "tab:blue", "y_offset_0p5mm": "tab:orange"}
    labels = {"on_axis": "y0 = 0", "y_offset_0p5mm": "y0 = 0.5 mm"}
    for name, arrays in cases.items():
        color = colors[name]
        label = labels[name]
        time_ms = 1.0e3 * arrays["time_s"]
        axes[0, 0].plot(time_ms, 1.0e3 * arrays["position_m"][:, 0], color=color, label=label)
        axes[1, 0].plot(time_ms, arrays["velocity_m_per_s"][:, 0], color=color)
        axes[2, 0].plot(time_ms, 1.0e21 * arrays["force_n"][:, 0], color=color)
        axes[3, 0].plot(
            time_ms,
            1.0e-6 * arrays["sigma_plus_shift_hz"],
            color=color,
        )

        diag = diagnostics[name]
        diag_time_ms = 1.0e3 * diag["time_s"]
        axes[0, 1].plot(diag_time_ms, diag["total_intensity_w_per_m2"], color=color)
        axes[1, 1].plot(
            diag_time_ms,
            diag["x_beam_rates_per_s"][:, 0] - diag["x_beam_rates_per_s"][:, 1],
            color=color,
        )
        axes[2, 1].plot(
            diag_time_ms,
            1.0e19 * diag["position_force_gradient_x_n_per_m"],
            color=color,
        )
        axes[3, 1].plot(
            diag_time_ms,
            1.0e21 * diag["velocity_force_gradient_x_n_s_per_m"],
            color=color,
        )
    for focus_mm in (-10.0, 10.0):
        axes[0, 0].axhline(focus_mm, color="0.5", linewidth=0.8, linestyle=":")
    axes[0, 0].set_ylabel("x [mm]")
    axes[1, 0].set_ylabel(r"$v_x$ [m/s]")
    axes[2, 0].set_ylabel(r"$F_x$ [zN]")
    axes[3, 0].set_ylabel(r"$\Delta\nu_{AC,+}$ [MHz]")
    axes[3, 0].set_xlabel("time [ms]")
    axes[0, 1].set_ylabel(r"total trap intensity [W/m$^2$]")
    axes[0, 1].set_yscale("symlog", linthresh=1.0)
    axes[1, 1].set_ylabel(r"$R_{+x}-R_{-x}$ [s$^{-1}$]")
    axes[2, 1].set_ylabel(r"$\partial F_x/\partial x$ [$10^{-19}$ N/m]")
    axes[3, 1].set_ylabel(r"$\partial F_x/\partial v_x$ [$10^{-21}$ N s/m]")
    axes[3, 1].set_xlabel("time [ms]")
    axes[0, 0].legend(loc="best")
    for axis in axes.flat:
        axis.axhline(0.0, color="0.25", linewidth=0.7)
        axis.grid(alpha=0.2)
    figure.suptitle("Two-level pMOT focus-crossing trajectory diagnosis")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_axial_landscape(landscape, output_path):
    figure, axes = plt.subplots(4, 1, figsize=(12.5, 13.5), sharex=True)
    x_mm = 1.0e3 * landscape["x_m"]
    styles = ((0.0, "tab:blue", "y = 0"), (0.5, "tab:orange", "y = 0.5 mm"))
    for y_mm, color, label in styles:
        rest_key = f"y_{y_mm:g}mm_vx_0mps"
        moving_key = f"y_{y_mm:g}mm_vx_-10mps"
        axes[0].plot(
            x_mm,
            landscape[f"{rest_key}_total_intensity_w_per_m2"],
            color=color,
            label=label,
        )
        shifts = landscape[f"{rest_key}_transition_shifts_hz"]
        axes[1].plot(x_mm, 1.0e-6 * shifts[:, 2], color=color, label=f"{label}, q=+1")
        axes[1].plot(x_mm, 1.0e-6 * shifts[:, 0], color=color, linestyle="--", label=f"{label}, q=-1")
        axes[2].plot(
            x_mm,
            1.0e21 * landscape[f"{rest_key}_force_x_n"],
            color=color,
            label=f"{label}, v=0",
        )
        axes[2].plot(
            x_mm,
            1.0e21 * landscape[f"{moving_key}_force_x_n"],
            color=color,
            linestyle="--",
            label=f"{label}, v=-10 m/s",
        )
        axes[3].plot(
            x_mm,
            1.0e19 * landscape[f"{rest_key}_force_gradient_x_n_per_m"],
            color=color,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"trap intensity [W/m$^2$]")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("transition shift [MHz]")
    axes[2].set_ylabel(r"$F_x$ [zN]")
    axes[3].set_ylabel(r"$\partial F_x/\partial x$ [$10^{-19}$ N/m]")
    axes[3].set_xlabel("x [mm]")
    for axis in axes:
        axis.axhline(0.0, color="0.25", linewidth=0.7)
        axis.axvline(-10.0, color="0.5", linewidth=0.8, linestyle=":")
        axis.axvline(10.0, color="0.5", linewidth=0.8, linestyle=":")
        axis.axvline(0.0, color="0.4", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend(loc="best", ncol=2)
    figure.suptitle("Static axial landscape: narrow waists and limited restoring basin")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_beam_resolved(cases, diagnostics, output_path):
    figure, axes = plt.subplots(3, 2, figsize=(13.5, 10.5), sharex="col")
    for column, name in enumerate(("on_axis", "y_offset_0p5mm")):
        diag = diagnostics[name]
        time_ms = 1.0e3 * diag["time_s"]
        axes[0, column].plot(time_ms, diag["x_beam_rates_per_s"][:, 0], label="+x cooling")
        axes[0, column].plot(time_ms, diag["x_beam_rates_per_s"][:, 1], label="-x cooling")
        axes[1, column].plot(
            time_ms,
            1.0e-6 * diag["x_beam_closest_detuning_hz"][:, 0],
            label="+x closest allowed channel",
        )
        axes[1, column].plot(
            time_ms,
            1.0e-6 * diag["x_beam_closest_detuning_hz"][:, 1],
            label="-x closest allowed channel",
        )
        for q_index, q_label in enumerate(("q=-1", "q=0", "q=+1")):
            axes[2, column].plot(
                time_ms,
                1.0e-6 * diag["transition_shifts_hz"][:, q_index],
                label=q_label,
            )
        axes[0, column].set_title("on axis" if column == 0 else "y = 0.5 mm")
        axes[0, column].set_ylabel(r"x-beam rate [s$^{-1}$]")
        axes[1, column].set_ylabel("effective detuning [MHz]")
        axes[2, column].set_ylabel("AC Stark shift [MHz]")
        axes[2, column].set_xlabel("time [ms]")
        for row in range(3):
            axes[row, column].axhline(0.0, color="0.25", linewidth=0.7)
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend(loc="best")
    figure.suptitle("Beam-resolved resonance crossings along each trajectory")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    simple, cooling, trapping, table = _build_model()
    initial_states = {
        "on_axis": AtomState(
            position_m=(25.0e-3, 0.0, 0.0),
            velocity_m_per_s=(-10.0, 0.0, 0.0),
        ),
        "y_offset_0p5mm": AtomState(
            position_m=(25.0e-3, 0.5e-3, 0.0),
            velocity_m_per_s=(-10.0, 0.0, 0.0),
        ),
    }
    cases = {}
    diagnostics = {}
    summaries = {}
    for name, initial_state in initial_states.items():
        record = simulate_pmot_2level_trajectory(
            cooling,
            trapping,
            initial_state,
            DURATION_S,
            TIME_STEP_S,
            path_powers_w=PATH_POWERS_W,
            simple_config=simple,
            polarizability_table=table,
        )
        arrays = _record_arrays(record)
        diag = _trajectory_diagnostics(
            simple, cooling, trapping, table, arrays
        )
        cases[name] = arrays
        diagnostics[name] = diag
        summaries[name] = _summary(name, arrays, diag)
        np.savez_compressed(OUTPUT_ROOT / f"{name}_trajectory.npz", **arrays)
        np.savez_compressed(OUTPUT_ROOT / f"{name}_diagnostics.npz", **diag)

    landscape = _axial_landscape(simple, cooling, trapping, table)
    np.savez_compressed(OUTPUT_ROOT / "axial_landscape.npz", **landscape)
    convergence = {
        name: _short_convergence(simple, cooling, trapping, table, state)
        for name, state in initial_states.items()
    }

    _plot_trajectory_overview(
        cases, diagnostics, OUTPUT_ROOT / "trajectory_overview.png"
    )
    _plot_axial_landscape(landscape, OUTPUT_ROOT / "axial_landscape.png")
    _plot_beam_resolved(
        cases, diagnostics, OUTPUT_ROOT / "beam_resolved_diagnostics.png"
    )

    manifest = {
        "status": "PROVISIONAL_EFFECTIVE_TWO_LEVEL_PMOT_DIAGNOSTIC",
        "cooling_power_mw_per_beam": 1.0e3 * COOLING_POWER_W,
        "cooling_detuning_mhz": 1.0e-6 * COOLING_DETUNING_HZ,
        "trapping_path_powers_mw_xyz": (1.0e3 * PATH_POWERS_W).tolist(),
        "calibrated_effective_gradient_proxy_g_per_cm": [10.0, 10.0, -20.0],
        "duration_ms": 1.0e3 * DURATION_S,
        "time_step_us": 1.0e6 * TIME_STEP_S,
        "gravity_enabled": bool(simple.include_gravity),
        "trapping_waist_radius_um": float(1.0e6 * trapping[0].waist_radius_m),
        "trapping_rayleigh_range_um": float(1.0e6 * trapping[0].rayleigh_range_m),
        "trajectory_summaries": summaries,
        "short_5ms_timestep_refinement": convergence,
        "limitations": [
            "no conservative 1529-nm dipole force",
            "no trapping-light scattering or heating",
            "no coherent standing wave",
            "no m_F-resolved Stark Hamiltonian or optical pumping",
            "fallback/adiabatic quantization-axis prescription at vector zeros",
        ],
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
