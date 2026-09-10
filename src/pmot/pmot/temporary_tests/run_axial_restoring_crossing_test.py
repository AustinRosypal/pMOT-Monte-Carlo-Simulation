"""Distinguish axial pMOT restoration from ordinary optical damping.

This temporary diagnostic compares the same 25 m/s axial launch with the
provisional vector-only trapping shift enabled and disabled.  It also computes
the zero-velocity force along x so that a trajectory reversal is supported by
an explicit displacement-restoring force measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...mot_multilevel.rate_equations import RateEquationTrajectoryConfig
from ..trajectory_plotting import plot_pmot_trajectory_diagnostics
from ..vector_only_trajectories import build_vector_only_trajectory_context
from ..vector_only_trajectories import inward_launch_state
from ..vector_only_trajectories import simulate_vector_only_pmot_trajectory
from ..vector_only_trajectories import vector_only_trajectory_dataframe
from ..vector_only_trajectories import vector_only_trajectory_observable


CAMPAIGN_NAME = "temporary_axial_restoring_crossing_25mps_20260909"
LAUNCH_SPEED_M_PER_S = 25.0
DURATION_S = 10.0e-3
PRIMARY_TIME_STEP_S = 1.25e-6
COARSE_TIME_STEP_S = 2.5e-6


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _linear_crossings(times, values, companion=None):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    companion_values = None if companion is None else np.asarray(companion, dtype=float)
    records = []
    for index in np.flatnonzero(values[:-1] * values[1:] < 0.0):
        fraction = -values[index] / (values[index + 1] - values[index])
        record = {
            "time_ms": 1.0e3
            * float(times[index] + fraction * (times[index + 1] - times[index])),
            "index_before_crossing": int(index),
        }
        if companion_values is not None:
            record["interpolated_companion"] = float(
                companion_values[index]
                + fraction * (companion_values[index + 1] - companion_values[index])
            )
        records.append(record)
    return records


def _trajectory_summary(record):
    base = record.rate_equation
    time = np.asarray(base.times_s, dtype=float)
    position = np.asarray(base.positions_m, dtype=float)
    velocity = np.asarray(base.velocities_m_per_s, dtype=float)
    force = np.asarray(base.forces_n, dtype=float)
    x = position[:, 0]
    vx = velocity[:, 0]
    fx = force[:, 0]
    origin_crossings = _linear_crossings(time, x, vx)
    turns = _linear_crossings(time, vx, x)
    first_turn = turns[0] if turns else None
    if first_turn is not None:
        turn_index = first_turn["index_before_crossing"]
        fraction = -vx[turn_index] / (vx[turn_index + 1] - vx[turn_index])
        first_turn["interpolated_x_mm"] = 1.0e3 * first_turn.pop(
            "interpolated_companion"
        )
        first_turn["interpolated_force_n"] = float(
            fx[turn_index] + fraction * (fx[turn_index + 1] - fx[turn_index])
        )
        first_turn["force_times_displacement_j"] = float(
            first_turn["interpolated_force_n"]
            * first_turn["interpolated_x_mm"]
            * 1.0e-3
        )
    return {
        "termination_reason": base.termination_reason,
        "capture": asdict(record.capture),
        "sample_count": len(time),
        "elapsed_time_ms": 1.0e3 * time[-1],
        "origin_crossing_count": len(origin_crossings),
        "origin_crossings": origin_crossings,
        "turning_point_count": len(turns),
        "first_turning_point": first_turn,
        "minimum_x_mm": 1.0e3 * np.min(x),
        "final_x_mm": 1.0e3 * x[-1],
        "final_vx_m_per_s": vx[-1],
        "minimum_radius_mm": 1.0e3 * np.min(np.linalg.norm(position, axis=1)),
        "restoring_fraction_away_from_axis": float(
            np.mean((fx[np.abs(x) > 10.0e-6] * x[np.abs(x) > 10.0e-6]) < 0.0)
        ),
        "damping_fraction_while_moving": float(
            np.mean((fx[np.abs(vx) > 1.0e-3] * vx[np.abs(vx) > 1.0e-3]) < 0.0)
        ),
    }


def _static_force_lineout(enabled_context, disabled_context):
    x_mm = np.concatenate(
        (np.linspace(-2.0, -0.01, 200), np.linspace(0.01, 2.0, 200))
    )
    rows = []
    for coordinate_mm in x_mm:
        position = (1.0e-3 * coordinate_mm, 0.0, 0.0)
        previous_axis = (1.0 if coordinate_mm > 0.0 else -1.0, 0.0, 0.0)
        enabled = vector_only_trajectory_observable(
            enabled_context,
            position,
            (0.0, 0.0, 0.0),
            previous_axis,
        )
        disabled = vector_only_trajectory_observable(
            disabled_context,
            position,
            (0.0, 0.0, 0.0),
            previous_axis,
        )
        rows.append(
            {
                "x_mm": coordinate_mm,
                "force_x_trapping_on_n": enabled.rate_equation.force_n[0],
                "force_x_trapping_off_n": disabled.rate_equation.force_n[0],
                "effective_field_x_trapping_on_g": (
                    1.0e4 * enabled.stark_diagnostic.effective_field_t[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def _comparison_plot(enabled, disabled, lineout, path):
    enabled_base = enabled.rate_equation
    disabled_base = disabled.rate_equation
    figure, panels = plt.subplots(2, 2, figsize=(13.2, 8.6))
    styles = (
        (enabled_base, "38.294486 mW/path vector shift", "#2563eb"),
        (disabled_base, "zero trapping-power control", "#dc2626"),
    )
    for record, label, color in styles:
        time_ms = 1.0e3 * np.asarray(record.times_s)
        position_mm = 1.0e3 * np.asarray(record.positions_m)[:, 0]
        velocity = np.asarray(record.velocities_m_per_s)[:, 0]
        panels[0, 0].plot(time_ms, position_mm, color=color, label=label)
        panels[0, 1].plot(time_ms, velocity, color=color, label=label)
        panels[1, 0].plot(position_mm, velocity, color=color, label=label)
    panels[0, 0].axhline(0.0, color="black", linewidth=0.8)
    panels[0, 0].axhspan(-2.0, 2.0, color="#16a34a", alpha=0.08)
    panels[0, 0].set(title="Axial position", xlabel="Time [ms]", ylabel="x [mm]")
    panels[0, 1].axhline(0.0, color="black", linewidth=0.8)
    panels[0, 1].set(title="Axial velocity", xlabel="Time [ms]", ylabel=r"$v_x$ [m/s]")
    panels[1, 0].axhline(0.0, color="black", linewidth=0.8)
    panels[1, 0].axvline(0.0, color="black", linewidth=0.8)
    panels[1, 0].set(title="Axial phase space", xlabel="x [mm]", ylabel=r"$v_x$ [m/s]")
    panels[1, 1].plot(
        lineout["x_mm"],
        1.0e21 * lineout["force_x_trapping_on_n"],
        color="#2563eb",
        label="trapping shift on",
    )
    panels[1, 1].plot(
        lineout["x_mm"],
        1.0e21 * lineout["force_x_trapping_off_n"],
        color="#dc2626",
        label="trapping shift off",
    )
    panels[1, 1].axhline(0.0, color="black", linewidth=0.8)
    panels[1, 1].axvline(0.0, color="black", linewidth=0.8)
    panels[1, 1].set(
        title="Zero-velocity force lineout",
        xlabel="x [mm]",
        ylabel=r"$F_x$ [$10^{-21}$ N]",
    )
    for panel in panels.flat:
        panel.grid(alpha=0.22)
        panel.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "25 m/s axial pMOT crossing and displacement-restoration test\n"
        "vector-only stretched-transition diagnostic; gravity on; diffusion off"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run() -> dict:
    root = _project_root() / "outputs" / "diagnostics" / "pmot" / CAMPAIGN_NAME
    data_root = root / "data"
    figure_root = root / "figures"
    data_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    enabled_context = build_vector_only_trajectory_context()
    disabled_context = build_vector_only_trajectory_context(
        trapping_power_w_per_path=0.0
    )
    initial = inward_launch_state(speed_m_per_s=LAUNCH_SPEED_M_PER_S)
    primary_config = RateEquationTrajectoryConfig(
        time_step_s=PRIMARY_TIME_STEP_S,
        include_diffusion=False,
        escape_radius_m=30.0e-3,
    )
    coarse_config = RateEquationTrajectoryConfig(
        time_step_s=COARSE_TIME_STEP_S,
        include_diffusion=False,
        escape_radius_m=30.0e-3,
    )

    print("[1/4] 25 m/s trajectory with vector trapping shift", flush=True)
    enabled = simulate_vector_only_pmot_trajectory(
        initial,
        duration_s=DURATION_S,
        context=enabled_context,
        trajectory_config=primary_config,
    )
    print("[2/4] matched zero-trapping-power control", flush=True)
    disabled = simulate_vector_only_pmot_trajectory(
        initial,
        duration_s=DURATION_S,
        context=disabled_context,
        trajectory_config=primary_config,
    )
    print("[3/4] 2.5 us convergence comparison", flush=True)
    coarse = simulate_vector_only_pmot_trajectory(
        initial,
        duration_s=DURATION_S,
        context=enabled_context,
        trajectory_config=coarse_config,
    )
    print("[4/4] static zero-velocity force lineout and figures", flush=True)
    lineout = _static_force_lineout(enabled_context, disabled_context)

    vector_only_trajectory_dataframe(enabled).to_csv(
        data_root / "trajectory_25mps_trapping_on.csv", index=False
    )
    vector_only_trajectory_dataframe(disabled).to_csv(
        data_root / "trajectory_25mps_trapping_off_control.csv", index=False
    )
    lineout.to_csv(data_root / "zero_velocity_force_lineout_x.csv", index=False)
    _comparison_plot(
        enabled,
        disabled,
        lineout,
        figure_root / "axial_crossing_restoration_vs_damping_control.png",
    )
    figure, _ = plot_pmot_trajectory_diagnostics(
        enabled,
        list(enabled_context.cooling_repump_beams),
        list(enabled_context.trapping_beams),
        path=figure_root / "axial_25mps_trajectory_and_beams.png",
        title="25 m/s axial pMOT trajectory with local effective-field axis",
        axial_extent_m=27.0e-3,
    )
    plt.close(figure)

    enabled_summary = _trajectory_summary(enabled)
    disabled_summary = _trajectory_summary(disabled)
    coarse_summary = _trajectory_summary(coarse)
    slope_mask = np.abs(lineout["x_mm"]) <= 0.25
    slope_n_per_m = float(
        np.polyfit(
            1.0e-3 * lineout.loc[slope_mask, "x_mm"],
            lineout.loc[slope_mask, "force_x_trapping_on_n"],
            1,
        )[0]
    )
    restoring = bool(
        np.all(
            lineout["x_mm"].to_numpy()
            * lineout["force_x_trapping_on_n"].to_numpy()
            < 0.0
        )
    )
    summary = {
        "status": "PROVISIONAL_VECTOR_ONLY_AXIAL_RESTORATION_DIAGNOSTIC",
        "launch_speed_m_per_s": LAUNCH_SPEED_M_PER_S,
        "launch_position_mm": [15.0, 0.0, 0.0],
        "duration_ms": 1.0e3 * DURATION_S,
        "primary_time_step_us": 1.0e6 * PRIMARY_TIME_STEP_S,
        "coarse_time_step_us": 1.0e6 * COARSE_TIME_STEP_S,
        "trapping_power_mw_per_path": (
            1.0e3 * enabled_context.trapping_power_w_per_path
        ),
        "enabled": enabled_summary,
        "zero_trapping_power_control": disabled_summary,
        "coarse_enabled": coarse_summary,
        "classification_matches_coarse": (
            enabled.capture.classification == coarse.capture.classification
        ),
        "first_turn_x_difference_primary_vs_coarse_mm": abs(
            enabled_summary["first_turning_point"]["interpolated_x_mm"]
            - coarse_summary["first_turning_point"]["interpolated_x_mm"]
        ),
        "zero_velocity_force_is_restoring_at_every_nonzero_lineout_point": restoring,
        "central_force_slope_n_per_m": slope_n_per_m,
        "interpretation": (
            "The enabled trajectory crosses x=0, reverses at negative x, and "
            "returns. The matched zero-power atom stops at positive x. Together "
            "with Fx*x<0 on the zero-velocity lineout, this separates position "
            "restoration from velocity damping in this provisional model."
        ),
        "limitations": [
            "ideal scalar/tensor cancellation is imposed",
            "stretched-transition differential vector polarizability proxy",
            "first-order trajectory integrator",
            "focal peak is not temporally resolved",
            "recoil diffusion and 1529-nm heating are absent",
            "one-dimensional restoration is not three-dimensional stability",
        ],
    }
    (root / "summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# 25 m/s axial crossing and restoration diagnostic\n\n"
        "This temporary test distinguishes velocity damping from displacement "
        "restoration by comparing a vector-shift trajectory with an otherwise "
        "identical zero-trapping-power control. See `summary.json` for the "
        "quantitative crossing and turning-point results. This is a provisional "
        "vector-only stretched-transition calculation, not a validated pMOT "
        "capture claim.\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_ready(summary), indent=2), flush=True)
    return summary


if __name__ == "__main__":
    run()
