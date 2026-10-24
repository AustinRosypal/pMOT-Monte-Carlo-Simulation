"""Extend the focused two-level pMOT launch to 50 ms and diagnose drift.

This is a provisional effective-two-level diagnostic, not a physical pMOT
prediction.  The gravity-off control separates optical dynamics from vertical
free fall, while the on-axis launch is retained as a focus-crossing reference.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagnose_pmot_2level_focus_trajectory import (
    PATH_POWERS_W,
    _build_model,
    _evaluate,
    _record_arrays,
)
from pmot.configuration import RB87_MASS_KG
from pmot.pMOT_2Level import simulate_pmot_2level_trajectory
from pmot.state import AtomState


OUTPUT_ROOT = Path(
    "outputs/diagnostics/pMOT_2Level/focus_trajectory_50ms_20261016"
)
DURATION_S = 50.0e-3
TIME_STEP_S = 2.5e-6
DIAGNOSTIC_STRIDE = 20


def _sample_diagnostics(simple, cooling, trapping, table, arrays):
    indices = np.arange(0, arrays["time_s"].size, DIAGNOSTIC_STRIDE)
    if indices[-1] != arrays["time_s"].size - 1:
        indices = np.append(indices, arrays["time_s"].size - 1)

    force = arrays["force_n"][indices]
    position = arrays["position_m"][indices]
    velocity = arrays["velocity_m_per_s"][indices]
    radius = np.linalg.norm(position, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    radial_force = np.zeros_like(radius)
    nonzero = radius > 0.0
    radial_force[nonzero] = np.einsum(
        "ij,ij->i", force[nonzero], position[nonzero]
    ) / radius[nonzero]
    optical_power = np.einsum("ij,ij->i", force, velocity)

    transition_shifts = np.empty((indices.size, 3))
    scattering_rate = np.empty(indices.size)
    total_intensity = np.empty(indices.size)
    axis_defined = np.empty(indices.size, dtype=bool)
    for output_index, record_index in enumerate(indices):
        observable = _evaluate(
            simple,
            cooling,
            trapping,
            table,
            arrays["position_m"][record_index],
            arrays["velocity_m_per_s"][record_index],
            arrays["quantization_axis"][record_index],
        )
        transition_shifts[output_index] = observable.stark.transition_shifts_hz_by_q
        scattering_rate[output_index] = observable.total_scattering_rate_per_s
        total_intensity[output_index] = np.sum(
            observable.stark.component_intensities_w_per_m2
        )
        axis_defined[output_index] = (
            observable.stark.quantization_axis_from_vector_shift
        )

    gravity = (
        np.asarray(simple.gravity_acceleration_m_per_s2, dtype=float)
        if simple.include_gravity
        else np.zeros(3)
    )
    total_acceleration = force / RB87_MASS_KG + gravity
    radial_acceleration = np.zeros_like(radius)
    radial_acceleration[nonzero] = np.einsum(
        "ij,ij->i", total_acceleration[nonzero], position[nonzero]
    ) / radius[nonzero]
    return {
        "index": indices,
        "time_s": arrays["time_s"][indices],
        "position_m": position,
        "velocity_m_per_s": velocity,
        "force_n": force,
        "radius_m": radius,
        "speed_m_per_s": speed,
        "radial_optical_force_n": radial_force,
        "optical_power_w": optical_power,
        "total_acceleration_m_per_s2": total_acceleration,
        "radial_total_acceleration_m_per_s2": radial_acceleration,
        "transition_shifts_hz": transition_shifts,
        "total_scattering_rate_per_s": scattering_rate,
        "total_intensity_w_per_m2": total_intensity,
        "axis_defined_by_vector_shift": axis_defined,
    }


def _nearest_index(time_s, target_s):
    return int(np.argmin(np.abs(time_s - target_s)))


def _case_summary(name, simple, arrays, diagnostics):
    requested_times_ms = (0.0, 25.0, 30.0, 40.0, 50.0)
    snapshots = {}
    for time_ms in requested_times_ms:
        index = _nearest_index(arrays["time_s"], time_ms * 1.0e-3)
        diagnostic_index = _nearest_index(
            diagnostics["time_s"], arrays["time_s"][index]
        )
        snapshots[f"{time_ms:g}_ms"] = {
            "position_mm": (1.0e3 * arrays["position_m"][index]).tolist(),
            "radius_mm": float(
                1.0e3 * np.linalg.norm(arrays["position_m"][index])
            ),
            "velocity_m_per_s": arrays["velocity_m_per_s"][index].tolist(),
            "speed_m_per_s": float(
                np.linalg.norm(arrays["velocity_m_per_s"][index])
            ),
            "optical_force_zN": (
                1.0e21 * arrays["force_n"][index]
            ).tolist(),
            "radial_total_acceleration_m_per_s2": float(
                diagnostics["radial_total_acceleration_m_per_s2"][diagnostic_index]
            ),
            "optical_power_f_dot_v_fW": float(
                1.0e15 * diagnostics["optical_power_w"][diagnostic_index]
            ),
        }

    after_25 = arrays["time_s"] >= 25.0e-3
    radius_after_25 = np.linalg.norm(arrays["position_m"][after_25], axis=1)
    position_after_25 = arrays["position_m"][after_25]
    velocity_after_25 = arrays["velocity_m_per_s"][after_25]
    diag_after_25 = diagnostics["time_s"] >= 25.0e-3
    return {
        "case": name,
        "gravity_enabled": bool(simple.include_gravity),
        "snapshots": snapshots,
        "radius_range_after_25ms_mm": [
            float(1.0e3 * np.min(radius_after_25)),
            float(1.0e3 * np.max(radius_after_25)),
        ],
        "coordinate_range_after_25ms_mm": {
            axis: [
                float(1.0e3 * np.min(position_after_25[:, axis_index])),
                float(1.0e3 * np.max(position_after_25[:, axis_index])),
            ]
            for axis_index, axis in enumerate("xyz")
        },
        "velocity_range_after_25ms_m_per_s": {
            axis: [
                float(np.min(velocity_after_25[:, axis_index])),
                float(np.max(velocity_after_25[:, axis_index])),
            ]
            for axis_index, axis in enumerate("xyz")
        },
        "fraction_after_25ms_optically_damping": float(
            np.mean(diagnostics["optical_power_w"][diag_after_25] < 0.0)
        ),
        "fraction_after_25ms_total_acceleration_radially_inward": float(
            np.mean(
                diagnostics["radial_total_acceleration_m_per_s2"][diag_after_25]
                < 0.0
            )
        ),
        "maximum_radius_time_ms": float(
            1.0e3
            * arrays["time_s"][after_25][int(np.argmax(radius_after_25))]
        ),
    }


def _plot_offset_comparison(cases, diagnostics, output_path):
    figure, axes = plt.subplots(3, 2, figsize=(14.0, 11.0), sharex=True)
    style = {
        "offset_gravity_on": ("tab:blue", "gravity on"),
        "offset_gravity_off": ("tab:orange", "gravity off"),
    }
    component_colors = ("tab:red", "tab:green", "tab:purple")
    for name in ("offset_gravity_on", "offset_gravity_off"):
        arrays = cases[name]
        diag = diagnostics[name]
        color, label = style[name]
        time_ms = 1.0e3 * arrays["time_s"]
        diag_time_ms = 1.0e3 * diag["time_s"]
        axes[0, 0].plot(
            time_ms,
            1.0e3 * np.linalg.norm(arrays["position_m"], axis=1),
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            time_ms,
            np.linalg.norm(arrays["velocity_m_per_s"], axis=1),
            color=color,
            label=label,
        )
        axes[2, 0].plot(
            diag_time_ms,
            diag["radial_total_acceleration_m_per_s2"],
            color=color,
            label=label,
        )
        axes[2, 1].plot(
            diag_time_ms,
            1.0e15 * diag["optical_power_w"],
            color=color,
            label=label,
        )

    gravity_arrays = cases["offset_gravity_on"]
    time_ms = 1.0e3 * gravity_arrays["time_s"]
    for component, (axis_label, color) in enumerate(
        zip("xyz", component_colors, strict=True)
    ):
        axes[1, 0].plot(
            time_ms,
            1.0e3 * gravity_arrays["position_m"][:, component],
            color=color,
            label=axis_label,
        )
        axes[1, 1].plot(
            time_ms,
            gravity_arrays["velocity_m_per_s"][:, component],
            color=color,
            label=f"v{axis_label}",
        )

    for axis in axes.flat:
        axis.axvline(25.0, color="0.25", linestyle="--", linewidth=1.0)
        axis.axhline(0.0, color="0.35", linewidth=0.7)
        axis.grid(alpha=0.22)
        axis.legend(loc="best")
    axes[0, 0].set_ylabel("distance from origin [mm]")
    axes[0, 1].set_ylabel("speed [m/s]")
    axes[1, 0].set_ylabel("gravity-on position [mm]")
    axes[1, 1].set_ylabel("gravity-on velocity [m/s]")
    axes[2, 0].set_ylabel("radial total acceleration [m/s²]\n(negative = inward)")
    axes[2, 1].set_ylabel(r"optical $\mathbf{F}\!\cdot\!\mathbf{v}$ [fW]\n(negative = damping)")
    axes[2, 0].set_xlabel("time [ms]")
    axes[2, 1].set_xlabel("time [ms]")
    figure.suptitle(
        "50-ms offset launch: outward drift is not simply free fall\n"
        "provisional effective two-level pMOT"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_case_overview(cases, output_path):
    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    labels = {
        "offset_gravity_on": "offset, gravity on",
        "offset_gravity_off": "offset, gravity off",
        "on_axis_gravity_on": "on axis, gravity on",
        "on_axis_gravity_off": "on axis, gravity off",
    }
    colors = {
        "offset_gravity_on": "tab:blue",
        "offset_gravity_off": "tab:orange",
        "on_axis_gravity_on": "tab:green",
        "on_axis_gravity_off": "tab:red",
    }
    for name, arrays in cases.items():
        time_ms = 1.0e3 * arrays["time_s"]
        axes[0].plot(
            time_ms,
            1.0e3 * np.linalg.norm(arrays["position_m"], axis=1),
            label=labels[name],
            color=colors[name],
        )
        axes[1].plot(
            time_ms,
            np.linalg.norm(arrays["velocity_m_per_s"], axis=1),
            label=labels[name],
            color=colors[name],
        )
    for axis in axes:
        axis.axvline(25.0, color="0.25", linestyle="--", linewidth=1.0)
        axis.grid(alpha=0.22)
        axis.legend(loc="best")
    axes[0].set_ylabel("distance from origin [mm]")
    axes[1].set_ylabel("speed [m/s]")
    axes[1].set_xlabel("time [ms]")
    figure.suptitle("50-ms comparison of the focus-crossing launches")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    simple_gravity, cooling, trapping, table = _build_model()
    simple_no_gravity = replace(simple_gravity, include_gravity=False)
    offset = AtomState(
        position_m=(25.0e-3, 0.5e-3, 0.0),
        velocity_m_per_s=(-10.0, 0.0, 0.0),
    )
    on_axis = AtomState(
        position_m=(25.0e-3, 0.0, 0.0),
        velocity_m_per_s=(-10.0, 0.0, 0.0),
    )
    specifications = {
        "offset_gravity_on": (simple_gravity, offset),
        "offset_gravity_off": (simple_no_gravity, offset),
        "on_axis_gravity_on": (simple_gravity, on_axis),
        "on_axis_gravity_off": (simple_no_gravity, on_axis),
    }
    cases = {}
    diagnostics = {}
    summaries = {}
    for name, (simple, initial_state) in specifications.items():
        trajectory_path = OUTPUT_ROOT / f"{name}_trajectory.npz"
        diagnostic_path = OUTPUT_ROOT / f"{name}_diagnostics.npz"
        if trajectory_path.exists() and diagnostic_path.exists():
            print(f"loading {name}", flush=True)
            with np.load(trajectory_path) as stored:
                arrays = {key: stored[key] for key in stored.files}
            with np.load(diagnostic_path) as stored:
                diagnostic = {key: stored[key] for key in stored.files}
        else:
            print(f"running {name}", flush=True)
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
            diagnostic = _sample_diagnostics(
                simple, cooling, trapping, table, arrays
            )
            np.savez_compressed(trajectory_path, **arrays)
            np.savez_compressed(diagnostic_path, **diagnostic)
        cases[name] = arrays
        diagnostics[name] = diagnostic
        summaries[name] = _case_summary(name, simple, arrays, diagnostic)

    _plot_offset_comparison(
        cases, diagnostics, OUTPUT_ROOT / "offset_50ms_drift.png"
    )
    _plot_case_overview(cases, OUTPUT_ROOT / "all_cases_50ms.png")
    manifest = {
        "status": "PROVISIONAL_EFFECTIVE_TWO_LEVEL_PMOT_DIAGNOSTIC",
        "duration_ms": 1.0e3 * DURATION_S,
        "time_step_us": 1.0e6 * TIME_STEP_S,
        "cooling_power_mw_per_beam": 27.0,
        "cooling_detuning_mhz": -15.0,
        "trapping_path_powers_mw_xyz": (1.0e3 * PATH_POWERS_W).tolist(),
        "gravity_force_zN": float(
            1.0e21
            * RB87_MASS_KG
            * simple_gravity.gravity_acceleration_m_per_s2[2]
        ),
        "cases": summaries,
        "limitations": [
            "transition-level scalar/vector/tensor Stark ansatz",
            "history-dependent quantization-axis fallback at vector zeros",
            "no conservative 1529-nm dipole force",
            "no trapping-light scattering or heating",
            "no m_F-resolved optical pumping",
            "no coherent standing-wave structure",
        ],
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
