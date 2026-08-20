"""Legacy photon-event performance records, plots, and command-line runner.

The efficient main solver is :mod:`pmot.mot_multilevel.rate_equations`. This
module remains for short event-driven regression and visualization runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..mot.magnetic_fields import default_anti_helmholtz_config
from .atomic_structure import AtomicStructure, build_atomic_structure
from .configuration import (
    DarkStateBehavior,
    InitializationMode,
    MultilevelMOTConfig,
    default_multilevel_mot_config,
    multilevel_mot_paths,
)
from .screening import CORE_RADIUS_M, core_entry_count, draw_multilevel_mot_beam_volumes
from .simulation import (
    MultilevelTrajectoryRecord,
    build_multilevel_mot_beams,
    simulate_multilevel_trajectory,
)
from .trajectory import MultilevelAtomState, sample_initial_internal_state


AXIS_COLORS = ("#b91c1c", "#1d4ed8", "#15803d")
DEFAULT_LAUNCHES = (
    ((8.0e-3, 0.0, 0.0), (-2.0, 0.0, 0.0)),
    ((0.0, 8.0e-3, 0.0), (0.0, -4.0, 0.2)),
    ((0.0, 0.0, 8.0e-3), (0.2, 0.0, -4.0)),
    ((6.0e-3, 6.0e-3, 0.0), (-3.0, -3.0, 0.15)),
    ((-6.0e-3, 0.0, 6.0e-3), (3.0, 0.15, -3.0)),
    ((4.0e-3, -5.0e-3, 6.0e-3), (-2.0, 2.5, -3.0)),
)


@dataclass(frozen=True, slots=True)
class MOTPerformanceConfig:
    """Controls a small, inspectable set of full-history MOT trajectories."""

    trajectory_count: int = 3
    duration_s: float = 5.0e-3
    max_events: int = 100_000
    escape_radius_m: float = 30.0e-3
    seed: int = 20260819
    repumper_enabled: bool = True
    initialization_mode: InitializationMode = InitializationMode.VAPOR


def _validate_record(record: MultilevelTrajectoryRecord, beam_count: int) -> None:
    lengths = {
        len(record.times_s),
        len(record.positions_m),
        len(record.velocities_m_per_s),
        len(record.mean_forces_n),
        len(record.total_scattering_rates_per_s),
        len(record.beam_scattering_rates_per_s),
        len(record.magnetic_fields_t),
        len(record.internal_state_indices),
        len(record.event_types),
        len(record.event_beam_indices),
    }
    if len(lengths) != 1 or not record.times_s:
        raise ValueError("trajectory record histories must be nonempty and aligned")
    if np.any(np.diff(record.times_s) < 0.0):
        raise ValueError("trajectory record times must be nondecreasing")
    if any(len(rates) != beam_count for rates in record.beam_scattering_rates_per_s):
        raise ValueError("each per-beam rate row must match the beam count")


def _save_figure(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def _display_indices(length: int, maximum: int = 25_000) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum, dtype=int))


def _time_scale(record: MultilevelTrajectoryRecord) -> tuple[float, str]:
    elapsed = record.times_s[-1] - record.times_s[0]
    return (1e3, "ms") if elapsed >= 1e-3 else (1e6, "µs")


def save_trajectory_data(
    record: MultilevelTrajectoryRecord,
    structure: AtomicStructure,
    beams,
    output_stem: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> list[Path]:
    """Save the complete event history in NPZ and human-readable CSV formats."""

    _validate_record(record, len(beams))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    states = [structure.states[index] for index in record.internal_state_indices]
    beam_labels = np.asarray([beam.label for beam in beams], dtype=str)
    families = np.asarray([beam.family for beam in beams], dtype=str)
    npz_path = output_stem.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        time_s=np.asarray(record.times_s),
        position_m=np.asarray(record.positions_m),
        velocity_m_per_s=np.asarray(record.velocities_m_per_s),
        conditional_mean_force_n=np.asarray(record.mean_forces_n),
        total_available_absorption_rate_per_s=np.asarray(record.total_scattering_rates_per_s),
        beam_available_absorption_rate_per_s=np.asarray(record.beam_scattering_rates_per_s),
        magnetic_field_t=np.asarray(record.magnetic_fields_t),
        internal_state_index=np.asarray(record.internal_state_indices, dtype=int),
        manifold=np.asarray([state.manifold for state in states], dtype=str),
        F=np.asarray([state.f for state in states], dtype=int),
        mF=np.asarray([state.m_f for state in states], dtype=int),
        event_type=np.asarray(record.event_types, dtype=str),
        event_beam_index=np.asarray([-1 if i is None else i for i in record.event_beam_indices], dtype=int),
        beam_label=beam_labels,
        beam_family=families,
    )

    csv_path = output_stem.with_suffix(".csv")
    rate_columns = [f"available_absorption_rate_{index}_per_s" for index in range(len(beams))]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s", "x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s",
            "conditional_Fx_N", "conditional_Fy_N", "conditional_Fz_N",
            "total_available_absorption_rate_per_s", *rate_columns,
            "Bx_T", "By_T", "Bz_T", "state_index", "manifold", "F", "mF", "event", "event_beam_index",
        ))
        for index, state in enumerate(states):
            writer.writerow((
                record.times_s[index], *record.positions_m[index], *record.velocities_m_per_s[index],
                *record.mean_forces_n[index], record.total_scattering_rates_per_s[index],
                *record.beam_scattering_rates_per_s[index], *record.magnetic_fields_t[index],
                record.internal_state_indices[index], state.manifold, state.f, state.m_f,
                record.event_types[index], record.event_beam_indices[index],
            ))

    metadata_path = output_stem.with_name(f"{output_stem.name}_metadata.json")
    payload = {
        "schema": "pmot.multilevel.performance.v1",
        "rate_definition": (
            "Instantaneous available laser-absorption hazard evaluated in the definite internal state; "
            "it is not a binned count of realized photon events."
        ),
        "beam_columns": [
            {
                "column": rate_columns[index],
                "index": index,
                "label": beam.label,
                "family": beam.family,
                "direction": list(beam.direction),
                "power_W": beam.power_w,
                "detuning_Hz": beam.detuning_hz,
            }
            for index, beam in enumerate(beams)
        ],
        "termination_reason": record.termination_reason,
        "counters": asdict(record.counters),
        **(metadata or {}),
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return [npz_path, csv_path, metadata_path]


def plot_cartesian_kinematics(record: MultilevelTrajectoryRecord, path: Path, title: str) -> Path:
    """Plot position and velocity separately for each Cartesian component."""

    scale, unit = _time_scale(record)
    indices = _display_indices(len(record.times_s))
    time = scale * np.asarray(record.times_s)[indices]
    position_mm = 1e3 * np.asarray(record.positions_m)[indices]
    velocity = np.asarray(record.velocities_m_per_s)[indices]
    figure, panels = plt.subplots(2, 3, figsize=(15, 7.5), sharex=True, constrained_layout=True)
    for component, label in enumerate("xyz"):
        panels[0, component].plot(time, position_mm[:, component], color=AXIS_COLORS[component])
        panels[1, component].plot(time, velocity[:, component], color=AXIS_COLORS[component])
        panels[0, component].set(title=rf"${label}(t)$", ylabel="Position [mm]")
        panels[1, component].set(title=rf"$v_{label}(t)$", xlabel=f"Time [{unit}]", ylabel="Velocity [m/s]")
        for panel in panels[:, component]:
            panel.axhline(0.0, color="#64748b", linewidth=.7)
            panel.grid(alpha=.22)
    figure.suptitle(title)
    return _save_figure(figure, path)


def plot_internal_state_history(
    record: MultilevelTrajectoryRecord,
    structure: AtomicStructure,
    path: Path,
    title: str,
) -> Path:
    """Plot hyperfine manifold, mF, and the categorical definite state."""

    scale, unit = _time_scale(record)
    indices = _display_indices(len(record.times_s))
    time = scale * np.asarray(record.times_s)[indices]
    states = [structure.states[record.internal_state_indices[index]] for index in indices]
    f_values = np.asarray([state.f for state in states])
    m_values = np.asarray([state.m_f for state in states])
    labels = [f"{state.manifold[0]} F={state.f}, mF={state.m_f:+d}" for state in states]
    ordered = list(dict.fromkeys(labels))
    codes = np.asarray([ordered.index(label) for label in labels])
    figure, panels = plt.subplots(
        3, 1, figsize=(14, max(8.0, 4.0 + .24 * len(ordered))), sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, max(1.5, .16 * len(ordered)))}, constrained_layout=True,
    )
    panels[0].step(time, f_values, where="post", color="#7c3aed")
    panels[0].set(ylabel="F", yticks=(0, 1, 2, 3), title="Hyperfine manifold")
    panels[1].step(time, m_values, where="post", color="#0f766e")
    panels[1].set(ylabel=r"$m_F$", title="Zeeman sublevel")
    panels[2].step(time, codes, where="post", color="#c2410c", linewidth=1.4)
    panels[2].set_yticks(range(len(ordered)), ordered)
    panels[2].set(xlabel=f"Time [{unit}]", ylabel="Definite state", title="Complete internal-state walk")
    for panel in panels:
        panel.grid(alpha=.2, axis="x")
    figure.suptitle(title)
    return _save_figure(figure, path)


def plot_scattering_rates(record: MultilevelTrajectoryRecord, beams, path: Path, title: str) -> Path:
    """Plot total, family-summed, and individual available absorption rates."""

    scale, unit = _time_scale(record)
    indices = _display_indices(len(record.times_s))
    time = scale * np.asarray(record.times_s)[indices]
    rates = np.asarray(record.beam_scattering_rates_per_s)[indices]
    total = np.asarray(record.total_scattering_rates_per_s)[indices]
    families = list(dict.fromkeys(beam.family for beam in beams))
    figure, panels = plt.subplots(1 + len(families), 1, figsize=(14, 4.0 + 3.2 * len(families)), sharex=True, constrained_layout=True)
    panels = np.atleast_1d(panels)
    panels[0].step(time, total, where="post", color="#111827", label="all optical components")
    for family, color in zip(families, ("#2563eb", "#c2410c", "#7c3aed")):
        columns = [index for index, beam in enumerate(beams) if beam.family == family]
        panels[0].step(time, np.sum(rates[:, columns], axis=1), where="post", color=color, alpha=.85, label=family)
    panels[0].set(title="Total and optical-family rates", ylabel=r"Rate [s$^{-1}$]")
    panels[0].legend(ncol=max(1, len(families) + 1), fontsize=8)
    for panel, family in zip(panels[1:], families):
        for beam_index, beam in enumerate(beams):
            if beam.family == family:
                direction = "".join(f"{value:+.0f}" for value in beam.direction)
                panel.step(time, rates[:, beam_index], where="post", linewidth=1.0, label=f"{beam_index}: {beam.axis_name} k={direction}")
        panel.set(title=f"Per-beam {family} absorption rates", ylabel=r"Rate [s$^{-1}$]")
        panel.legend(ncol=3, fontsize=7)
    panels[-1].set_xlabel(f"Time [{unit}]")
    for panel in panels:
        panel.grid(alpha=.2)
    figure.suptitle(title + "\nAvailable absorption hazards, not binned event counts")
    return _save_figure(figure, path)


def plot_radial_performance(record: MultilevelTrajectoryRecord, path: Path, title: str) -> Path:
    """Plot radius and speed, the simplest single-atom cooling/trapping view."""

    scale, unit = _time_scale(record)
    indices = _display_indices(len(record.times_s))
    time = scale * np.asarray(record.times_s)[indices]
    radius_mm = 1e3 * np.linalg.norm(np.asarray(record.positions_m)[indices], axis=1)
    speed = np.linalg.norm(np.asarray(record.velocities_m_per_s)[indices], axis=1)
    figure, panels = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    panels[0].plot(time, radius_mm, color="#7c3aed")
    panels[0].axhline(1e3 * CORE_RADIUS_M, color="#15803d", linestyle="--", label="2 mm diagnostic core")
    panels[0].set(ylabel="Radius [mm]", title="Distance from trap center")
    panels[0].legend()
    panels[1].plot(time, speed, color="#0f766e")
    panels[1].set(xlabel=f"Time [{unit}]", ylabel="Speed [m/s]", title="Atomic speed")
    for panel in panels:
        panel.grid(alpha=.22)
    figure.suptitle(title)
    return _save_figure(figure, path)


def plot_trajectory_3d(record: MultilevelTrajectoryRecord, beams, path: Path, title: str) -> Path:
    """Plot the atom path, trap center, diagnostic core, and beam volumes."""

    indices = _display_indices(len(record.times_s))
    position_mm = 1e3 * np.asarray(record.positions_m)[indices]
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    draw_multilevel_mot_beam_volumes(axis, beams)
    axis.plot(*position_mm.T, color="#1d4ed8", linewidth=1.6, label="atom")
    axis.scatter(*position_mm[0], color="#dc2626", s=45, label="start")
    axis.scatter(*position_mm[-1], color="#111827", s=45, label="end")
    axis.scatter(0.0, 0.0, 0.0, color="#15803d", s=45, label="trap center")
    extent = max(10.0, 1.08 * float(np.max(np.abs(position_mm))))
    axis.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent), xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]", title=title)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.legend()
    return _save_figure(figure, path)


def trajectory_performance_summary(record: MultilevelTrajectoryRecord) -> dict[str, object]:
    positions = np.asarray(record.positions_m)
    velocities = np.asarray(record.velocities_m_per_s)
    radii = np.linalg.norm(positions, axis=1)
    speeds = np.linalg.norm(velocities, axis=1)
    entries = core_entry_count(record)
    bounded = bounded_trajectory_diagnostics(record)
    return {
        "termination_reason": record.termination_reason,
        "completed_requested_duration": record.termination_reason in {"duration", "duration_dark_ballistic"},
        "elapsed_s": float(record.times_s[-1] - record.times_s[0]),
        "record_count": len(record.times_s),
        "initial_position_m": positions[0].tolist(),
        "final_position_m": positions[-1].tolist(),
        "initial_velocity_m_per_s": velocities[0].tolist(),
        "final_velocity_m_per_s": velocities[-1].tolist(),
        "initial_radius_m": float(radii[0]),
        "minimum_radius_m": float(np.min(radii)),
        "final_radius_m": float(radii[-1]),
        "initial_speed_m_per_s": float(speeds[0]),
        "minimum_speed_m_per_s": float(np.min(speeds)),
        "final_speed_m_per_s": float(speeds[-1]),
        "core_entry_count": entries,
        "two_core_entry_candidate": entries >= 2,
        "bounded_trajectory_diagnostic": bounded,
        "counters": asdict(record.counters),
    }


def bounded_trajectory_diagnostics(
    record: MultilevelTrajectoryRecord,
    *,
    bounded_radius_m: float = CORE_RADIUS_M,
    final_window_s: float = 1.0e-3,
    minimum_completed_duration_s: float = 5.0e-3,
) -> dict[str, object]:
    """Conservative post-run check that the atom stays central at late times.

    This is a convergence diagnostic, not a universal definition of capture.
    A credible trapping result must remain stable when the duration and final
    window are increased and the event cap is raised.
    """

    times = np.asarray(record.times_s, dtype=float)
    positions = np.asarray(record.positions_m, dtype=float)
    elapsed_s = float(times[-1] - times[0])
    completed = record.termination_reason in {"duration", "duration_dark_ballistic"}
    window_duration_s = min(final_window_s, elapsed_s)
    window_start_s = times[-1] - window_duration_s
    sample_times = np.linspace(window_start_s, times[-1], 501)
    sampled_position = np.column_stack([
        np.interp(sample_times, times, positions[:, component]) for component in range(3)
    ])
    window_radii = np.linalg.norm(sampled_position, axis=1)
    candidate = bool(
        completed
        and elapsed_s >= minimum_completed_duration_s
        and np.max(window_radii) <= bounded_radius_m
    )
    return {
        "candidate_bounded_trapped": candidate,
        "completed_requested_duration": completed,
        "minimum_required_duration_s": minimum_completed_duration_s,
        "final_window_s": window_duration_s,
        "bounded_radius_m": bounded_radius_m,
        "final_window_maximum_radius_m": float(np.max(window_radii)),
        "final_window_rms_radius_m": float(np.sqrt(np.mean(window_radii**2))),
        "interpretation": (
            "Candidate only; repeat with longer duration/window and a nonbinding event cap."
        ),
    }


def generate_performance_bundle(
    record: MultilevelTrajectoryRecord,
    structure: AtomicStructure,
    beams,
    output_directory: Path,
    trajectory_index: int,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Save all requested numerical histories and performance figures."""

    _validate_record(record, len(beams))
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = output_directory / f"trajectory_{trajectory_index:03d}"
    title = f"Full multilevel MOT trajectory {trajectory_index:03d}"
    files = save_trajectory_data(record, structure, beams, stem, metadata=metadata)
    files += [
        plot_cartesian_kinematics(record, stem.with_name(f"{stem.name}_cartesian.png"), title),
        plot_internal_state_history(record, structure, stem.with_name(f"{stem.name}_internal_state.png"), title),
        plot_scattering_rates(record, beams, stem.with_name(f"{stem.name}_scattering_rates.png"), title),
        plot_radial_performance(record, stem.with_name(f"{stem.name}_radius_speed.png"), title),
        plot_trajectory_3d(record, beams, stem.with_name(f"{stem.name}_3d.png"), title),
    ]
    summary = trajectory_performance_summary(record)
    summary["outputs"] = [str(path) for path in files]
    return summary


def run_mot_performance(
    run_config: MOTPerformanceConfig | None = None,
    *,
    output_directory: Path | None = None,
    config: MultilevelMOTConfig | None = None,
    coil_config=None,
) -> dict[str, object]:
    """Run representative moving atoms through the full cooling+repump MOT."""

    run = run_config or MOTPerformanceConfig()
    if run.trajectory_count <= 0 or run.duration_s <= 0.0 or run.max_events <= 0 or run.escape_radius_m <= 0.0:
        raise ValueError("trajectory_count, duration_s, max_events, and escape_radius_m must be positive")
    cfg = replace(
        config or default_multilevel_mot_config(),
        repumper_enabled=run.repumper_enabled,
        initialization_mode=run.initialization_mode,
        dark_state_behavior=DarkStateBehavior.BALLISTIC,
        diagnostics_enabled=True,
    )
    structure = build_atomic_structure()
    beams = build_multilevel_mot_beams(config=cfg)
    coil = coil_config or default_anti_helmholtz_config()
    output = output_directory or multilevel_mot_paths()["trajectories"] / "mot_performance"
    output.mkdir(parents=True, exist_ok=True)
    initial_rng = np.random.default_rng(run.seed)
    runs: list[dict[str, object]] = []
    for index in range(run.trajectory_count):
        position, velocity = DEFAULT_LAUNCHES[index % len(DEFAULT_LAUNCHES)]
        state_index = sample_initial_internal_state(structure, cfg.initialization_mode, initial_rng)
        initial = MultilevelAtomState(position, velocity, state_index)
        print(
            f"[MOT performance] running trajectory {index + 1}/{run.trajectory_count} "
            f"(seed={run.seed + index}, max_events={run.max_events})",
            flush=True,
        )
        record = simulate_multilevel_trajectory(
            initial, run.duration_s, coil, beams=beams, structure=structure, config=cfg,
            seed=run.seed + index, max_events=run.max_events, escape_radius_m=run.escape_radius_m,
        )
        internal = structure.states[state_index]
        metadata = {
            "trajectory_index": index,
            "seed": run.seed + index,
            "requested_duration_s": run.duration_s,
            "max_events": run.max_events,
            "repumper_enabled": cfg.repumper_enabled,
            "initialization_mode": str(cfg.initialization_mode),
            "initial_internal_state": {"index": state_index, "F": internal.f, "mF": internal.m_f},
            "multilevel_config": asdict(cfg),
        }
        result = generate_performance_bundle(record, structure, beams, output, index, metadata=metadata)
        result.update({"trajectory_index": index, "seed": run.seed + index, "initial_F": internal.f, "initial_mF": internal.m_f})
        runs.append(result)
        print(
            f"[MOT performance] completed trajectory {index + 1}/{run.trajectory_count}: "
            f"termination={record.termination_reason}, records={len(record.times_s)}",
            flush=True,
        )

    summary = {
        "schema": "pmot.multilevel.performance-summary.v1",
        "run_config": asdict(run),
        "repumper_enabled": cfg.repumper_enabled,
        "warning": (
            "Individual trajectories are diagnostics, not proof of cooling or trapping. "
            "Require ensembles and convergence under longer duration and larger max_events."
        ),
        "event_cap_terminations": sum(item["termination_reason"] == "max_events" for item in runs),
        "two_core_entry_candidate_count": sum(bool(item["two_core_entry_candidate"]) for item in runs),
        "bounded_trapped_candidate_count": sum(
            bool(item["bounded_trajectory_diagnostic"]["candidate_bounded_trapped"]) for item in runs
        ),
        "trajectories": runs,
    }
    summary_path = output / "mot_performance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    defaults = MOTPerformanceConfig()
    parser = argparse.ArgumentParser(description="Generate full multilevel MOT trajectory performance diagnostics")
    parser.add_argument("--trajectory-count", type=int, default=defaults.trajectory_count)
    parser.add_argument("--duration-ms", type=float, default=1e3 * defaults.duration_s)
    parser.add_argument("--max-events", type=int, default=defaults.max_events)
    parser.add_argument("--escape-radius-mm", type=float, default=1e3 * defaults.escape_radius_m)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repumper", action=argparse.BooleanOptionalAction, default=defaults.repumper_enabled)
    parser.add_argument("--initialization", choices=[mode.value for mode in InitializationMode], default=defaults.initialization_mode.value)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run = MOTPerformanceConfig(
        trajectory_count=args.trajectory_count,
        duration_s=1e-3 * args.duration_ms,
        max_events=args.max_events,
        escape_radius_m=1e-3 * args.escape_radius_mm,
        seed=args.seed,
        repumper_enabled=args.repumper,
        initialization_mode=InitializationMode(args.initialization),
    )
    print(json.dumps(run_mot_performance(run, output_directory=args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
