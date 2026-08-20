"""Small, deliberately pre-statistical multilevel capture screening runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..mot.magnetic_fields import default_anti_helmholtz_config
from .atomic_structure import AtomicStructure, build_atomic_structure
from .configuration import DarkStateBehavior, default_multilevel_mot_config, multilevel_mot_paths
from .simulation import MultilevelTrajectoryRecord, build_multilevel_cooling_beams, simulate_multilevel_trajectory
from .trajectory import MultilevelAtomState


CORE_RADIUS_M = 2.0e-3


def resampled_radii_m(record: MultilevelTrajectoryRecord, spatial_resolution_m: float = 0.2e-3) -> np.ndarray:
    """Resolve core crossings, including long ballistic segments after dark entry."""

    pieces: list[np.ndarray] = []
    positions = np.asarray(record.positions_m)
    for start, stop in zip(positions[:-1], positions[1:]):
        subdivisions = max(1, int(np.ceil(np.linalg.norm(stop - start) / spatial_resolution_m)))
        fraction = np.linspace(0.0, 1.0, subdivisions + 1, endpoint=True)
        segment = start[None, :] + fraction[:, None] * (stop - start)[None, :]
        pieces.append(np.linalg.norm(segment[:-1], axis=1))
    pieces.append(np.asarray([np.linalg.norm(positions[-1])]))
    return np.concatenate(pieces)


def core_entry_count(record: MultilevelTrajectoryRecord, core_radius_m: float = CORE_RADIUS_M) -> int:
    inside = resampled_radii_m(record) <= core_radius_m
    return int(np.count_nonzero(inside & np.concatenate(([True], ~inside[:-1]))))


def capture_classification(record: MultilevelTrajectoryRecord, *, repumper_enabled: bool = False) -> str:
    """Classify a completed trace without hiding incomplete event-capped runs."""

    if record.termination_reason == "max_events":
        return "indeterminate_event_cap"
    if record.termination_reason == "escaped":
        return "escaped"
    if core_entry_count(record) >= 2:
        return "candidate_trapped_two_core_entries"
    if record.counters.dark_entry_time_s is not None and not repumper_enabled:
        return "untrapped_dark"
    return "indeterminate_duration"


def trajectory_lifetime_s(record: MultilevelTrajectoryRecord) -> float:
    """Return the dark-entry lifetime when present, otherwise elapsed record time."""

    if record.counters.dark_entry_time_s is not None:
        return record.counters.dark_entry_time_s
    return record.times_s[-1] if record.times_s else 0.0


def lifetime_file_tag(record: MultilevelTrajectoryRecord) -> str:
    """Return a compact SI-labeled lifetime tag for filenames."""

    lifetime_us = 1e6 * trajectory_lifetime_s(record)
    return f"lifetime_{lifetime_us:.3f}us".replace(".", "p")


def draw_multilevel_mot_beam_volumes(axis, beams, length_m: float = 120.0e-3) -> None:
    """Draw multilevel MOT cooling-beam volumes with the simple-MOT notebook style."""

    from ..mot_simple.plotting import _beam_surface_mesh_mm

    axis_color = {
        "horizontal_x": "#f9a8d4",
        "horizontal_y": "#93c5fd",
        "vertical_z": "#86efac",
    }
    drawn_axes: set[str] = set()
    for beam in beams:
        if beam.axis_name in drawn_axes:
            continue
        drawn_axes.add(beam.axis_name)
        x_surface_mm, y_surface_mm, z_surface_mm = _beam_surface_mesh_mm(
            direction=beam.direction,
            radius_m=beam.beam_radius_m,
            length_m=length_m,
        )
        axis.plot_surface(
            x_surface_mm,
            y_surface_mm,
            z_surface_mm,
            color=axis_color[beam.axis_name],
            linewidth=0.0,
            antialiased=True,
            shade=False,
            alpha=0.22,
        )


def plot_screening_trajectory(record, structure, path: Path, index: int, beams=None) -> Path:
    import matplotlib.pyplot as plt

    times_ms = 1e3 * np.asarray(record.times_s)
    positions_mm = 1e3 * np.asarray(record.positions_m)
    speeds = np.linalg.norm(np.asarray(record.velocities_m_per_s), axis=1)
    radii_mm = np.linalg.norm(positions_mm, axis=1)
    f_values = np.asarray([structure.states[i].f for i in record.internal_state_indices])
    figure = plt.figure(figsize=(12, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    radial = figure.add_subplot(grid[0, 0]); speed = figure.add_subplot(grid[0, 1])
    path_axis = figure.add_subplot(grid[1, 0], projection="3d"); internal = figure.add_subplot(grid[1, 1])
    radial.plot(times_ms, radii_mm, color="#7c3aed"); radial.axhspan(0, 2, color="#86efac", alpha=.35, label="2 mm core")
    radial.set(xlabel="Time [ms]", ylabel="Radius [mm]", title="Distance from MOT center"); radial.legend(); radial.grid(alpha=.2)
    speed.plot(times_ms, speeds, color="#0f766e"); speed.set(xlabel="Time [ms]", ylabel="Speed [m/s]", title="Speed"); speed.grid(alpha=.2)
    if beams is not None:
        draw_multilevel_mot_beam_volumes(path_axis, beams)
    path_axis.scatter([0.0], [0.0], [0.0], color="#111827", s=34, label="trap center")
    path_axis.plot(*positions_mm.T, color="#2563eb"); path_axis.scatter(*positions_mm[0], color="#dc2626", label="launch"); path_axis.scatter(*positions_mm[-1], color="#111827", label="end")
    local_extent_mm = max(25.0, 1.08 * float(np.max(np.abs(positions_mm)))) if len(positions_mm) else 25.0
    path_axis.set(xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]", title="3D path with cooling beam volumes")
    path_axis.set_xlim(-local_extent_mm, local_extent_mm)
    path_axis.set_ylim(-local_extent_mm, local_extent_mm)
    path_axis.set_zlim(-local_extent_mm, local_extent_mm)
    path_axis.set_box_aspect((1.0, 1.0, 1.0))
    path_axis.legend()
    internal.step(times_ms, f_values, where="post", color="#c2410c"); internal.set_yticks((1, 2, 3)); internal.set(xlabel="Time [ms]", ylabel="F", title="Hyperfine manifold"); internal.grid(alpha=.2)
    figure.suptitle(f"Capture-screen trajectory {index:02d}: {capture_classification(record)}; core entries={core_entry_count(record)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170); plt.close(figure)
    return path


def beam_rate_column(beams, axis_name: str, source_side: str) -> int:
    """Return the beam index for a beam entering from one signed axis side."""

    axis_component = {"horizontal_x": 0, "horizontal_y": 1, "vertical_z": 2}[axis_name]
    desired_sign = -1.0 if source_side == "+" else 1.0
    for index, beam in enumerate(beams):
        if beam.axis_name == axis_name and np.sign(beam.direction[axis_component]) == desired_sign:
            return index
    raise ValueError(f"no {source_side}{axis_name[-1]}-side beam found")


def plot_velocity_and_beam_rates(record, beams, path: Path, index: int) -> Path:
    """Plot velocity components and signed-side cooling-beam scattering rates."""

    import matplotlib.pyplot as plt

    times_ms = 1e3 * np.asarray(record.times_s)
    velocities = np.asarray(record.velocities_m_per_s)
    beam_rates = np.asarray(record.beam_scattering_rates_per_s)
    axes_info = (("horizontal_x", "x", 0), ("horizontal_y", "y", 1), ("vertical_z", "z", 2))
    figure, panels = plt.subplots(3, 3, figsize=(13.5, 9.0), sharex=True, constrained_layout=True)
    for column, (axis_name, label, component) in enumerate(axes_info):
        panels[0, column].plot(times_ms, velocities[:, component], color="#0f766e", linewidth=1.8)
        panels[0, column].set(title=rf"$v_{label}(t)$", ylabel="m/s")
        for row, source_side, color in ((1, "+", "#b91c1c"), (2, "-", "#1d4ed8")):
            beam_index = beam_rate_column(beams, axis_name, source_side)
            propagation = "+" if beams[beam_index].direction[component] > 0.0 else "-"
            panels[row, column].plot(times_ms, beam_rates[:, beam_index], color=color, linewidth=1.8)
            panels[row, column].set(
                title=rf"{source_side}{label}-side beam, {propagation}{label} propagation",
                ylabel=r"s$^{-1}$",
            )
        panels[2, column].set_xlabel("Time [ms]")
    for panel in panels.flat:
        panel.grid(alpha=.22)
    figure.suptitle(
        f"Capture-screen trajectory {index:02d}: velocity and signed-side scattering rates\n"
        "Beam labels name the side the light comes from; rates are instantaneous absorption rates."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path


def hyperfine_walk_frame_indices(
    record,
    structure: AtomicStructure,
    max_transition_frames: int = 200,
    final_tail_frames: int = 35,
) -> list[int]:
    """Return displayed record indices while preserving the actual dark-entry tail."""

    event_indices = [i for i, event in enumerate(record.event_types) if event not in ("initial", "dark_ballistic_end", "duration")]
    displayed_events = event_indices[:max_transition_frames]
    if record.counters.dark_entry_time_s is not None:
        dark_index = next(i for i, state_index in enumerate(record.internal_state_indices) if structure.states[state_index].is_dark)
        dark_event_position = event_indices.index(dark_index)
        tail_start = max(0, dark_event_position - final_tail_frames)
        displayed_events = [*displayed_events, *event_indices[tail_start:dark_event_position + 1]]
    frame_indices = []
    for index in [0, *displayed_events]:
        if index not in frame_indices:
            frame_indices.append(index)
    if record.counters.dark_entry_time_s is not None:
        frame_indices.extend([dark_index] * 18)
    return frame_indices


def animate_hyperfine_walk(record, structure: AtomicStructure, path: Path, max_transition_frames: int = 200, fps: int = 6) -> Path:
    """Animate an actual stochastic walk through the Rb-87 hyperfine graph."""

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    frame_indices = hyperfine_walk_frame_indices(record, structure, max_transition_frames=max_transition_frames)

    level_y = {
        ("ground", 1): 0.0,
        ("ground", 2): 1.0,
        ("excited", 0): 2.3,
        ("excited", 1): 3.0,
        ("excited", 2): 4.0,
        ("excited", 3): 5.0,
    }
    figure, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)

    def draw(frame_number: int):
        axis.clear()
        for (manifold, f), y in level_y.items():
            color = "#94a3b8" if manifold == "ground" else "#fbbf24"
            axis.hlines(y, -f - .25, f + .25, color=color, linewidth=2)
            for m_f in range(-f, f + 1):
                axis.scatter(m_f, y, s=30, facecolor="white", edgecolor=color, zorder=2)
            axis.text(-3.75, y, f"{manifold[0]} F={f}", va="center", fontsize=10)
        record_index = frame_indices[frame_number]
        current = structure.states[record.internal_state_indices[record_index]]
        x_now, y_now = current.m_f, level_y[(current.manifold, current.f)]
        trail_start = max(1, frame_number - 10)
        for trail_frame in range(trail_start, frame_number + 1):
            if frame_indices[trail_frame] != frame_indices[trail_frame - 1] + 1:
                continue
            previous = structure.states[record.internal_state_indices[frame_indices[trail_frame - 1]]]
            destination = structure.states[record.internal_state_indices[frame_indices[trail_frame]]]
            axis.annotate("", xy=(destination.m_f, level_y[(destination.manifold, destination.f)]), xytext=(previous.m_f, level_y[(previous.manifold, previous.f)]), arrowprops={"arrowstyle": "->", "color": "#64748b", "alpha": .25 + .07 * (trail_frame - trail_start)})
        color = "#111827" if current.is_dark else ("#2563eb" if current.is_ground else "#f97316")
        axis.scatter(x_now, y_now, s=280, color=color, edgecolor="white", linewidth=2, zorder=5)
        event = record.event_types[record_index]
        time_us = 1e6 * record.times_s[record_index]
        axis.set_title(f"Actual stochastic hyperfine walk\nt={time_us:.3f} µs | {event} | {current.manifold} F={current.f}, mF={current.m_f:+d}")
        axis.set(xlim=(-4.4, 3.6), ylim=(-.7, 5.7), xlabel=r"$m_F$", ylabel="Hyperfine manifold (schematic energy)")
        axis.set_yticks([]); axis.grid(alpha=.15, axis="x")
        axis.text(.99, .02, "Selection rule on every optical jump: ΔmF = 0, ±1", transform=axis.transAxes, ha="right")

    movie = animation.FuncAnimation(figure, draw, frames=len(frame_indices), interval=1000 / fps, repeat=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(path, writer=animation.PillowWriter(fps=fps), dpi=110)
    plt.close(figure)
    return path


def run_capture_screening(root: Path | None = None, seed: int = 31082026) -> dict[str, object]:
    from .diagnostics import _save_trajectory_csv

    paths = multilevel_mot_paths(root); output = paths["trajectories"] / "capture_screening"
    structure = build_atomic_structure(); beams = build_multilevel_cooling_beams(); coil = default_anti_helmholtz_config()
    config = replace(default_multilevel_mot_config(), dark_state_behavior=DarkStateBehavior.BALLISTIC)
    # Positions sample axes, diagonals, and unequal 3D directions. Velocities are
    # predominantly inward but span well below and near the simple-MOT capture scale.
    launches = (
        ((8, 0, 0), (-2, 0, 0)), ((-8, 0, 0), (4, .2, 0)),
        ((0, 8, 0), (0, -6, .2)), ((0, -8, 0), (.2, 8, 0)),
        ((0, 0, 8), (0, 0, -2)), ((0, 0, -8), (0, 0, 4)),
        ((6, 6, 0), (-3, -3, .2)), ((-6, 0, 6), (4, .2, -4)),
        ((4, -5, 6), (-3, 3.75, -4.5)), ((-7, -3, -4), (6, 2.6, 3.4)),
    )
    summaries, records, animation_paths = [], [], []
    for index, (position_mm, velocity) in enumerate(launches):
        position = tuple(1e-3 * value for value in position_mm)
        m_f = (-2, -1, 0, 1, 2)[index % 5]
        initial = MultilevelAtomState(position, velocity, structure.state_index("ground", 2, m_f))
        record = simulate_multilevel_trajectory(
            initial, 5e-3, coil, beams=beams, structure=structure, config=config,
            seed=seed + index, max_events=5_000, escape_radius_m=30.0e-3,
        )
        records.append(record); classification = capture_classification(record)
        plot_screening_trajectory(record, structure, output / f"trajectory_{index:02d}.png", index, beams)
        plot_velocity_and_beam_rates(record, beams, output / f"trajectory_{index:02d}_velocity_and_beam_rates.png", index)
        gif_path = animate_hyperfine_walk(
            record,
            structure,
            output / f"hyperfine_walk_trajectory_{index:02d}_{lifetime_file_tag(record)}.gif",
        )
        animation_paths.append(gif_path)
        _save_trajectory_csv(record, structure, output / f"trajectory_{index:02d}.csv")
        summaries.append({"index": index, "initial_position_mm": position_mm, "initial_velocity_m_per_s": velocity, "initial_mF": m_f, "classification": classification, "core_entries": core_entry_count(record), "lifetime_us": float(1e6 * trajectory_lifetime_s(record)), "minimum_radius_mm": float(1e3 * np.min(resampled_radii_m(record))), "final_radius_mm": float(1e3 * np.linalg.norm(record.positions_m[-1])), "termination": record.termination_reason, **asdict(record.counters)})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "screening_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys()); writer.writeheader(); writer.writerows(summaries)
    result = {"seed": seed, "duration_ms": 5.0, "core_radius_mm": 2.0, "two_core_entry_candidate_count": sum(item["classification"] == "candidate_trapped_two_core_entries" for item in summaries), "dark_untrapped_count": sum(item["classification"] == "untrapped_dark" for item in summaries), "indeterminate_count": sum(item["classification"].startswith("indeterminate") for item in summaries), "escaped_count": sum(item["classification"] == "escaped" for item in summaries), "animations": [str(path) for path in animation_paths], "runs": summaries}
    (output / "screening_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    print(json.dumps(run_capture_screening(), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
