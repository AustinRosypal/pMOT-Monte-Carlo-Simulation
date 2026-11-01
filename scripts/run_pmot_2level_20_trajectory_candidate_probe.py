"""Run 20 provisional trajectories for the selected full two-level pMOT.

This is a deterministic transition-level diagnostic.  It is not a validated
capture cross section or loading-rate calculation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pMOT_2Level import simulate_pmot_2level_trajectory
from pmot.state import AtomState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "pMOT_2Level"
    / "candidate_6mW_m30MHz_46mW_20_trajectory_probe_20261028"
)
COOLING_POWER_MW = 6.0
COOLING_DETUNING_MHZ = -30.0
TRAPPING_POWER_MW_PER_PATH = 46.0
LAUNCH_DISTANCE_M = 7.0e-3
DURATION_S = 20.0e-3
COARSE_STEP_S = 10.0e-6
FINE_STEP_S = 5.0e-6
CORE_RADIUS_M = 2.0e-3
CORE_RESIDENCE_S = 5.0e-3
FINAL_RETENTION_S = 5.0e-3
FINE_AUDIT_INDICES = frozenset((1, 12, 13, 20))


def _unit(vector) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    return value / np.linalg.norm(value)


def _perpendicular(direction, preferred=(0.0, 0.0, 1.0)) -> np.ndarray:
    direction = _unit(direction)
    candidate = np.cross(direction, preferred)
    if np.linalg.norm(candidate) < 1.0e-12:
        candidate = np.cross(direction, (0.0, 1.0, 0.0))
    return _unit(candidate)


def _case(name, inward_direction, speed, impact_parameter_mm=0.0):
    inward = _unit(inward_direction)
    impact = 1.0e-3 * impact_parameter_mm * _perpendicular(inward)
    position = -LAUNCH_DISTANCE_M * inward + impact
    velocity = speed * inward
    return {
        "name": name,
        "position_m": position,
        "velocity_m_per_s": velocity,
        "speed_m_per_s": float(speed),
        "impact_parameter_mm": float(impact_parameter_mm),
    }


def _launch_cases():
    axes = (
        ("plus_x", (-1, 0, 0)),
        ("minus_x", (1, 0, 0)),
        ("plus_y", (0, -1, 0)),
        ("minus_y", (0, 1, 0)),
        ("plus_z", (0, 0, -1)),
        ("minus_z", (0, 0, 1)),
    )
    cases = []
    for speed in (8.0, 14.0):
        for label, inward in axes:
            cases.append(_case(f"{label}_{speed:g}mps", inward, speed))

    face_diagonals = (
        ("xy_pp", (-1, -1, 0)),
        ("xy_pm", (-1, 1, 0)),
        ("xz_pp", (-1, 0, -1)),
        ("yz_pm", (0, -1, 1)),
    )
    for label, inward in face_diagonals:
        cases.append(_case(f"face_{label}_10mps_b0p75", inward, 10.0, 0.75))

    body_diagonals = (
        ("ppp", (-1, -1, -1)),
        ("ppm", (-1, -1, 1)),
        ("pmp", (-1, 1, -1)),
        ("mpp", (1, -1, -1)),
    )
    for index, (label, inward) in enumerate(body_diagonals):
        speed = (6.0, 10.0, 14.0, 18.0)[index]
        cases.append(_case(f"body_{label}_{speed:g}mps_b1p5", inward, speed, 1.5))
    if len(cases) != 20:
        raise AssertionError(f"expected 20 cases, built {len(cases)}")
    return cases


def _arrays(record):
    return {
        "times_s": np.asarray(record.times_s),
        "positions_m": np.asarray(record.positions_m),
        "velocities_m_per_s": np.asarray(record.velocities_m_per_s),
        "forces_n": np.asarray(record.forces_n),
    }


def _core_summary(arrays) -> dict:
    times = arrays["times_s"]
    radius = np.linalg.norm(arrays["positions_m"], axis=1)
    inside = radius <= CORE_RADIUS_M
    entries = int(np.sum((~inside[:-1]) & inside[1:]))
    if inside[0]:
        entries += 1
    longest = 0.0
    start = None
    for index, is_inside in enumerate(inside):
        if is_inside and start is None:
            start = index
        if start is not None and (not is_inside or index == len(inside) - 1):
            end = index if is_inside else index - 1
            longest = max(longest, float(times[end] - times[start]))
            start = None
    final_mask = times >= times[-1] - FINAL_RETENTION_S - 1.0e-15
    final_window_inside = bool(np.all(inside[final_mask]))
    early = bool(longest >= CORE_RESIDENCE_S or entries >= 2)
    retained = bool(early and final_window_inside)
    return {
        "core_entry_count": entries,
        "longest_core_residence_ms": 1.0e3 * longest,
        "early_core_criterion": early,
        "final_5ms_inside_core": final_window_inside,
        "provisionally_retained": retained,
        "minimum_radius_mm": 1.0e3 * float(np.min(radius)),
        "final_radius_mm": 1.0e3 * float(radius[-1]),
        "final_speed_m_per_s": float(
            np.linalg.norm(arrays["velocities_m_per_s"][-1])
        ),
        "maximum_speed_m_per_s": float(
            np.max(np.linalg.norm(arrays["velocities_m_per_s"], axis=1))
        ),
    }


def _plot_trajectories(records, summaries, cases, path: Path) -> None:
    figure = plt.figure(figsize=(12.5, 9.5))
    axis = figure.add_subplot(111, projection="3d")
    theta = np.linspace(0.0, 2.0 * np.pi, 48)
    phi = np.linspace(0.0, np.pi, 24)
    sphere_x = 1.0e3 * CORE_RADIUS_M * np.outer(np.cos(theta), np.sin(phi))
    sphere_y = 1.0e3 * CORE_RADIUS_M * np.outer(np.sin(theta), np.sin(phi))
    sphere_z = 1.0e3 * CORE_RADIUS_M * np.outer(np.ones_like(theta), np.cos(phi))
    axis.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color="#60a5fa",
        alpha=0.10,
        linewidth=0.0,
    )
    for case, arrays, summary in zip(cases, records, summaries, strict=True):
        position_mm = 1.0e3 * arrays["positions_m"]
        color = "#15803d" if summary["provisionally_retained"] else "#b91c1c"
        axis.plot(*position_mm.T, color=color, linewidth=1.1, alpha=0.82)
        axis.scatter(*position_mm[0], color=color, marker="o", s=18)
        axis.scatter(*position_mm[-1], color=color, marker="x", s=22)
    axis.set_xlabel("x [mm]")
    axis.set_ylabel("y [mm]")
    axis.set_zlabel("z [mm]")
    axis.set_title(
        "20 Provisional Full 2-Level pMOT Trajectories\n"
        "green = retained for final 5 ms; red = not retained; blue = 2 mm core"
    )
    axis.set_box_aspect((1, 1, 1))
    all_positions_mm = np.concatenate(
        [1.0e3 * arrays["positions_m"] for arrays in records], axis=0
    )
    limit = max(8.0, min(30.0, 1.05 * float(np.max(np.abs(all_positions_mm)))))
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_radius(records, summaries, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    for arrays, summary in zip(records, summaries, strict=True):
        color = "#15803d" if summary["provisionally_retained"] else "#b91c1c"
        axis.plot(
            1.0e3 * arrays["times_s"],
            1.0e3 * np.linalg.norm(arrays["positions_m"], axis=1),
            color=color,
            linewidth=1.0,
            alpha=0.72,
        )
    axis.axhline(2.0, color="#2563eb", linestyle="--", linewidth=1.5, label="2 mm core")
    axis.axvspan(15.0, 20.0, color="#94a3b8", alpha=0.14, label="final 5 ms")
    axis.set_xlabel("time [ms]")
    axis.set_ylabel("radius from origin [mm]")
    axis.set_title("Radial Evolution of 20 Provisional pMOT Trajectories")
    axis.set_ylim(0.0, min(40.0, max(8.0, axis.get_ylim()[1])))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    base_apparatus = default_simple_mot_apparatus()
    apparatus = replace(
        base_apparatus,
        cooling=replace(
            base_apparatus.cooling,
            power_w_per_beam=1.0e-3 * COOLING_POWER_MW,
            detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        ),
    )
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        include_gravity=True,
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    trapping = build_pmot_2level_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    powers_w = (1.0e-3 * TRAPPING_POWER_MW_PER_PATH,) * 3
    cases = _launch_cases()

    primary_records = []
    primary_summaries = []
    rows = []
    for case_index, case in enumerate(cases, start=1):
        resolution = {}
        steps = [("coarse", COARSE_STEP_S)]
        if case_index in FINE_AUDIT_INDICES:
            steps.append(("fine", FINE_STEP_S))
        for label, step_s in steps:
            record = simulate_pmot_2level_trajectory(
                cooling,
                trapping,
                AtomState(
                    position_m=tuple(case["position_m"]),
                    velocity_m_per_s=tuple(case["velocity_m_per_s"]),
                ),
                DURATION_S,
                step_s,
                path_powers_w=powers_w,
                simple_config=simple,
                polarizability_table=table,
            )
            arrays = _arrays(record)
            summary = _core_summary(arrays)
            resolution[label] = {"arrays": arrays, "summary": summary}
            if label == "coarse":
                primary_records.append(arrays)
                primary_summaries.append(summary)
                np.savez_compressed(
                    OUTPUT_DIRECTORY / f"{case_index:02d}_{case['name']}.npz",
                    **arrays,
                )
        fine = resolution.get("fine")
        agreement = (
            resolution["coarse"]["summary"]["provisionally_retained"]
            == fine["summary"]["provisionally_retained"]
            if fine is not None
            else None
        )
        row = {
            "index": case_index,
            "name": case["name"],
            "initial_x_mm": 1.0e3 * case["position_m"][0],
            "initial_y_mm": 1.0e3 * case["position_m"][1],
            "initial_z_mm": 1.0e3 * case["position_m"][2],
            "initial_vx_m_per_s": case["velocity_m_per_s"][0],
            "initial_vy_m_per_s": case["velocity_m_per_s"][1],
            "initial_vz_m_per_s": case["velocity_m_per_s"][2],
            "initial_speed_m_per_s": case["speed_m_per_s"],
            "impact_parameter_mm": case["impact_parameter_mm"],
            **resolution["coarse"]["summary"],
            "fine_provisionally_retained": (
                fine["summary"]["provisionally_retained"]
                if fine is not None
                else None
            ),
            "classification_agrees_10us_5us": agreement,
            "fine_final_radius_mm": (
                fine["summary"]["final_radius_mm"] if fine is not None else None
            ),
            "fine_final_speed_m_per_s": (
                fine["summary"]["final_speed_m_per_s"]
                if fine is not None
                else None
            ),
        }
        rows.append(row)
        print(
            f"[{case_index:02d}/20] {case['name']}: "
            f"retained={row['provisionally_retained']}, "
            f"r_final={row['final_radius_mm']:.3f} mm, "
            f"dt_agrees={agreement}",
            flush=True,
        )

    summary_path = OUTPUT_DIRECTORY / "trajectory_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    trajectory_plot = OUTPUT_DIRECTORY / "trajectory_overview_3d.png"
    radius_plot = OUTPUT_DIRECTORY / "radius_vs_time.png"
    _plot_trajectories(primary_records, primary_summaries, cases, trajectory_plot)
    _plot_radius(primary_records, primary_summaries, radius_plot)

    retained_count = sum(row["provisionally_retained"] for row in rows)
    audited_rows = [
        row
        for row in rows
        if row["classification_agrees_10us_5us"] is not None
    ]
    agreement_count = sum(
        row["classification_agrees_10us_5us"] for row in audited_rows
    )
    manifest = {
        "status": "PROVISIONAL_TRANSITION_LEVEL_TRAJECTORY_DIAGNOSTIC",
        "warning": (
            "Not a validated pMOT capture or loading-rate prediction. The model "
            "omits conservative Stark-gradient force, 1529-nm scattering/heating, "
            "recoil diffusion, and nonadiabatic behavior at the vector-shift zero."
        ),
        "parameters": {
            "cooling_power_mw_per_component": COOLING_POWER_MW,
            "cooling_detuning_mhz": COOLING_DETUNING_MHZ,
            "trapping_power_mw_per_path_xyz": [TRAPPING_POWER_MW_PER_PATH] * 3,
            "launch_distance_mm": 1.0e3 * LAUNCH_DISTANCE_M,
            "duration_ms": 1.0e3 * DURATION_S,
            "gravity_included": True,
            "recoil_diffusion_included": False,
            "coarse_step_us": 1.0e6 * COARSE_STEP_S,
            "fine_step_us": 1.0e6 * FINE_STEP_S,
            "core_radius_mm": 1.0e3 * CORE_RADIUS_M,
            "core_residence_ms": 1.0e3 * CORE_RESIDENCE_S,
            "final_retention_window_ms": 1.0e3 * FINAL_RETENTION_S,
        },
        "launch_design": {
            "centered_axial": 12,
            "off_axis_face_diagonal": 4,
            "off_axis_body_diagonal": 4,
            "speeds_m_per_s": sorted({case["speed_m_per_s"] for case in cases}),
            "impact_parameters_mm": sorted(
                {case["impact_parameter_mm"] for case in cases}
            ),
        },
        "aggregate": {
            "trajectory_count": len(rows),
            "provisionally_retained_count": retained_count,
            "not_retained_count": len(rows) - retained_count,
            "classification_audit_count": len(audited_rows),
            "classification_agreement_count": agreement_count,
        },
        "rows": rows,
        "outputs": {
            "summary_csv": str(summary_path.relative_to(PROJECT_ROOT)),
            "trajectory_plot": str(trajectory_plot.relative_to(PROJECT_ROOT)),
            "radius_plot": str(radius_plot.relative_to(PROJECT_ROOT)),
        },
    }
    manifest_path = OUTPUT_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["aggregate"], indent=2), flush=True)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
