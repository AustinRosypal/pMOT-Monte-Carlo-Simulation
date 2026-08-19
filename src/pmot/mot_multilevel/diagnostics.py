"""Generate reproducible validation figures and tables for the multilevel MOT."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..mot.magnetic_fields import default_anti_helmholtz_config
from .atomic_structure import build_atomic_structure
from .configuration import DarkStateBehavior, default_multilevel_mot_config, multilevel_mot_paths
from .simulation import MultilevelTrajectoryRecord, build_multilevel_cooling_beams
from .simulation import build_multilevel_mot_beams
from .simulation import local_magnetic_field_t, simulate_multilevel_trajectory
from .simulation import unpolarized_f2_mean_observable
from .trajectory import MultilevelAtomState

COLORS = ("#b91c1c", "#1d4ed8", "#15803d")


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_force_curves(structure, beams, coil, config, path: Path) -> Path:
    positions = np.linspace(-8e-3, 8e-3, 121)
    velocities = np.linspace(-15.0, 15.0, 121)
    figure, panels = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    for component, label in enumerate("xyz"):
        restoring, damping = [], []
        for coordinate in positions:
            point = np.zeros(3); point[component] = coordinate
            restoring.append(unpolarized_f2_mean_observable(structure, beams, tuple(point), (0, 0, 0), coil, config).force_n[component])
        for speed in velocities:
            velocity = np.zeros(3); velocity[component] = speed
            damping.append(unpolarized_f2_mean_observable(structure, beams, (0, 0, 0), tuple(velocity), coil, config).force_n[component])
        restoring = np.asarray(restoring)
        # U(x) - U(0) = -integral_0^x F(x') dx'; integrate from the left,
        # then choose the MOT center as the zero of potential energy.
        potential = np.zeros_like(restoring)
        potential[1:] = -np.cumsum(0.5 * (restoring[1:] + restoring[:-1]) * np.diff(positions))
        potential -= potential[len(potential) // 2]
        panels[component, 0].plot(1e3 * positions, restoring, color="#7c3aed")
        panels[component, 1].plot(velocities, damping, color="#0f766e")
        panels[component, 2].plot(1e3 * positions, potential / 1.380649e-23 * 1e3, color="#c2410c")
        panels[component, 0].set(title=rf"$F_{label}({label})$, $v=0$", xlabel=f"{label} [mm]", ylabel=f"$F_{label}$ [N]")
        panels[component, 1].set(title=rf"$F_{label}(v_{label})$, $r=0$", xlabel=rf"$v_{label}$ [m/s]", ylabel=f"$F_{label}$ [N]")
        panels[component, 2].set(title=rf"$U_{label}({label})-U_{label}(0)$ from $F_{label}=-\partial_{{{label}}}U$", xlabel=f"{label} [mm]", ylabel=r"Potential [$k_B$ mK]")
        for panel in panels[component]:
            panel.axhline(0, color="0.4", lw=.8); panel.axvline(0, color="0.4", lw=.8); panel.grid(alpha=.25)
    figure.suptitle("Multilevel MOT instantaneous force: equal F=2 Zeeman populations\nRadiation pressure only; gravity excluded", fontsize=14)
    return _save(figure, path)


def plot_field_gradients(coil, path: Path) -> Path:
    coordinates = np.linspace(-10e-3, 10e-3, 161)
    figure, panels = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    for line_axis, panel in enumerate(panels):
        values = np.array([local_magnetic_field_t(tuple(c if i == line_axis else 0.0 for i in range(3)), coil) for c in coordinates]) * 1e4
        styles = ((0, ()), (0, (6, 2)), (0, (2, 2)))
        markers = ("o", "s", "^")
        for component, label in enumerate("xyz"):
            panel.plot(
                1e3 * coordinates, values[:, component], color=COLORS[component],
                linestyle=styles[component], linewidth=2.0 - .2 * component,
                marker=markers[component], markevery=(component * 4, 20), markersize=3.5,
                zorder=3 - component, label=f"$B_{label}$",
            )
        full_slope = np.polyfit(100 * coordinates, values[:, line_axis], 1)[0]
        center = len(coordinates) // 2
        local_slope = (values[center + 1, line_axis] - values[center - 1, line_axis]) / (100 * (coordinates[center + 1] - coordinates[center - 1]))
        panel.set(
            title=f"Along {'xyz'[line_axis]}: local={local_slope:.3f}, ±10 mm fit={full_slope:.3f} G/cm",
            xlabel=f"{'xyz'[line_axis]} [mm]", ylabel="Field [G]",
        )
        panel.grid(alpha=.25); panel.legend()
    figure.suptitle("Anti-Helmholtz magnetic-field component lineouts")
    return _save(figure, path)


def plot_trajectory(record: MultilevelTrajectoryRecord, structure, path: Path, title: str) -> Path:
    time_us = 1e6 * np.asarray(record.times_s); position_mm = 1e3 * np.asarray(record.positions_m)
    velocity = np.asarray(record.velocities_m_per_s); force = np.asarray(record.mean_forces_n)
    rate = np.asarray(record.total_scattering_rates_per_s)
    states = [structure.states[index] for index in record.internal_state_indices]
    state_labels = [f"{('g' if state.is_ground else 'e')} F={state.f}, m={state.m_f:+d}" for state in states]
    unique_labels = list(dict.fromkeys(state_labels))
    state_code = np.asarray([unique_labels.index(label) for label in state_labels])
    figure = plt.figure(figsize=(14, 12), constrained_layout=True); grid = figure.add_gridspec(3, 2)
    axes = [figure.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    trajectory_axis = figure.add_subplot(grid[2, 0], projection="3d"); state_axis = figure.add_subplot(grid[2, 1])
    for component, label in enumerate("xyz"):
        axes[0].plot(time_us, position_mm[:, component], color=COLORS[component], label=label)
        axes[1].plot(time_us, velocity[:, component], color=COLORS[component], label=f"v{label}")
        axes[3].plot(time_us, force[:, component], color=COLORS[component], label=f"F{label}")
    axes[2].plot(time_us, rate, color="#111827")
    axes[0].set(title="Position", ylabel="mm"); axes[1].set(title="Velocity", ylabel="m/s")
    axes[2].set(title="Available absorption rate", ylabel=r"s$^{-1}$"); axes[3].set(title="Conditional mean absorption force", ylabel="N")
    for axis in axes:
        axis.set_xlabel("Time [µs]"); axis.grid(alpha=.25)
    for axis in (axes[0], axes[1], axes[3]): axis.legend(loc="best")
    trajectory_axis.plot(position_mm[:, 0], position_mm[:, 1], position_mm[:, 2], color="#0f766e")
    trajectory_axis.scatter(*position_mm[0], color="#b91c1c", label="start"); trajectory_axis.scatter(*position_mm[-1], color="#111827", label="end")
    trajectory_axis.set(xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]", title="3D trajectory"); trajectory_axis.legend()
    state_axis.step(time_us, state_code, where="post", color="#c2410c", linewidth=1.8)
    for event_time, code, state in zip(time_us, state_code, states):
        state_axis.scatter(event_time, code, s=12, color="#111827" if state.is_ground else "#f59e0b", zorder=3)
    state_axis.set_yticks(range(len(unique_labels)), unique_labels)
    state_axis.set(title="Internal-state event timeline", xlabel="Time [µs]", ylabel="Hyperfine–Zeeman state"); state_axis.grid(alpha=.25, axis="x")
    figure.suptitle(f"{title}\ntermination={record.termination_reason}; absorptions={record.counters.absorption_events}; spontaneous={record.counters.spontaneous_emissions}")
    return _save(figure, path)


def plot_internal_state_timeline(record: MultilevelTrajectoryRecord, structure, path: Path, title: str) -> Path:
    """Draw a categorical dwell-time map of one atom's internal state."""

    times_us = 1e6 * np.asarray(record.times_s)
    states = [structure.states[index] for index in record.internal_state_indices]
    labels = [f"{('ground' if state.is_ground else 'excited')} F={state.f}, mF={state.m_f:+d}" for state in states]
    ordered_labels = list(dict.fromkeys(labels))
    y_values = [ordered_labels.index(label) for label in labels]
    figure, axis = plt.subplots(figsize=(13, max(4.5, .48 * len(ordered_labels))), constrained_layout=True)
    colors = {"bright": "#2563eb", "excited": "#f59e0b", "dark": "#111827"}
    for index in range(len(times_us) - 1):
        state = states[index]
        category = "dark" if state.is_dark else ("bright" if state.is_ground else "excited")
        axis.plot(times_us[index:index + 2], [y_values[index]] * 2, color=colors[category], linewidth=7, solid_capstyle="butt")
        if y_values[index + 1] != y_values[index]:
            axis.plot([times_us[index + 1]] * 2, y_values[index:index + 2], color="#64748b", linewidth=.7)
    axis.scatter(times_us[:-1], y_values[:-1], s=8, color="#0f172a", zorder=3, label="event")
    axis.set_yticks(range(len(ordered_labels)), ordered_labels)
    axis.set(xlabel="Time [µs]", ylabel="Definite internal state", title=title)
    axis.grid(alpha=.2, axis="x")
    axis.text(.99, .02, "blue: bright ground   amber: excited   black: dark F=1", transform=axis.transAxes, ha="right", va="bottom", fontsize=9)
    return _save(figure, path)


def _save_trajectory_csv(record, structure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    beam_count = max((len(rates) for rates in record.beam_scattering_rates_per_s), default=0)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s", "Fx_N", "Fy_N", "Fz_N", "absorption_rate_per_s", *(f"beam_{index}_absorption_rate_per_s" for index in range(beam_count)), "state_index", "manifold", "F", "mF", "event", "beam_index"))
        for i, state_index in enumerate(record.internal_state_indices):
            state = structure.states[state_index]
            beam_rates = tuple(record.beam_scattering_rates_per_s[i]) if i < len(record.beam_scattering_rates_per_s) else ()
            writer.writerow((record.times_s[i], *record.positions_m[i], *record.velocities_m_per_s[i], *record.mean_forces_n[i], record.total_scattering_rates_per_s[i], *beam_rates, state_index, state.manifold, state.f, state.m_f, record.event_types[i], record.event_beam_indices[i]))
    return path


def _save_repump_diagnostics(record: MultilevelTrajectoryRecord, path: Path) -> Path:
    """Save repump event and F=1 dwell diagnostics in analysis-friendly form."""

    path.parent.mkdir(parents=True, exist_ok=True)
    repump = record.repump_absorptions
    np.savez(
        path,
        repump_time_s=np.asarray([event.time_s for event in repump], dtype=float),
        repump_beam_index=np.asarray([event.beam_index for event in repump], dtype=int),
        repump_initial_f=np.asarray([event.initial_f for event in repump], dtype=int),
        repump_initial_m_f=np.asarray([event.initial_m_f for event in repump], dtype=int),
        repump_excited_f=np.asarray([event.excited_f for event in repump], dtype=int),
        repump_excited_m_f=np.asarray([event.excited_m_f for event in repump], dtype=int),
        repump_position_m=np.asarray([event.position_m for event in repump], dtype=float).reshape((-1, 3)),
        repump_velocity_m_per_s=np.asarray([event.velocity_m_per_s for event in repump], dtype=float).reshape((-1, 3)),
        repump_magnetic_field_t=np.asarray([event.magnetic_field_t for event in repump], dtype=float).reshape((-1, 3)),
        repump_detuning_rad_per_s=np.asarray([event.detuning_rad_per_s for event in repump], dtype=float),
        repump_polarization_weight=np.asarray([event.polarization_weight for event in repump], dtype=float),
        repump_saturation_parameter=np.asarray([event.saturation_parameter for event in repump], dtype=float),
        repump_rate_per_s=np.asarray([event.rate_per_s for event in repump], dtype=float),
        f1_visit_durations_s=np.asarray(record.f1_visit_durations_s, dtype=float),
        repump_photons_per_f1_return=np.asarray(record.repump_photons_per_f1_return, dtype=int),
        total_f1_time_s=np.asarray([record.counters.total_f1_time_s], dtype=float),
        f1_visit_count=np.asarray([record.counters.f1_visit_count], dtype=int),
        f1_returns_to_f2=np.asarray([record.counters.f1_returns_to_f2], dtype=int),
    )
    return path


def _save_repump_absorption_csv(record: MultilevelTrajectoryRecord, path: Path) -> Path:
    """Save one CSV row for each repump absorption."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "time_s", "beam_index", "initial_f", "initial_mF", "excited_f", "excited_mF",
            "x_m", "y_m", "z_m", "vx_m_per_s", "vy_m_per_s", "vz_m_per_s",
            "Bx_t", "By_t", "Bz_t", "detuning_rad_per_s", "polarization_weight",
            "saturation_parameter", "rate_per_s",
        ))
        for event in record.repump_absorptions:
            writer.writerow((
                event.time_s, event.beam_index, event.initial_f, event.initial_m_f,
                event.excited_f, event.excited_m_f, *event.position_m,
                *event.velocity_m_per_s, *event.magnetic_field_t,
                event.detuning_rad_per_s, event.polarization_weight,
                event.saturation_parameter, event.rate_per_s,
            ))
    return path


def plot_repump_diagnostics(records: list[MultilevelTrajectoryRecord], structure, beams, output_directory: Path) -> list[Path]:
    """Save first-pass repumper validation plots and numerical data."""

    output_directory.mkdir(parents=True, exist_ok=True)
    dwell_times_s = np.asarray([duration for record in records for duration in record.f1_visit_durations_s], dtype=float)
    photons_per_return = np.asarray([count for record in records for count in record.repump_photons_per_f1_return], dtype=int)
    repump_events = [event for record in records for event in record.repump_absorptions]
    beam_counts = {beam.label: 0 for beam in beams if beam.family == "repump"}
    channel_counts: dict[str, int] = {}
    for event in repump_events:
        beam_counts[beams[event.beam_index].label] += 1
        label = f"F=1,mF={event.initial_m_f:+d} -> F'={event.excited_f},mF'={event.excited_m_f:+d}"
        channel_counts[label] = channel_counts.get(label, 0) + 1
    total_f1 = sum(record.counters.total_f1_time_s for record in records)
    total_time = sum(record.times_s[-1] - record.times_s[0] for record in records if record.times_s)
    time_fractions = np.asarray([total_f1 / total_time if total_time > 0.0 else 0.0, 1.0 - total_f1 / total_time if total_time > 0.0 else 0.0])

    npz_path = output_directory / "repump_diagnostics.npz"
    np.savez(
        npz_path,
        f1_dwell_times_s=dwell_times_s,
        repump_photons_per_f1_return=photons_per_return,
        repump_beam_labels=np.asarray(list(beam_counts.keys()), dtype=str),
        repump_beam_counts=np.asarray(list(beam_counts.values()), dtype=int),
        repump_channel_labels=np.asarray(list(channel_counts.keys()), dtype=str),
        repump_channel_counts=np.asarray(list(channel_counts.values()), dtype=int),
        time_fraction_labels=np.asarray(["F=1", "not F=1"], dtype=str),
        time_fractions=time_fractions,
    )

    output_files = [npz_path]
    if len(dwell_times_s):
        figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        axis.hist(1e6 * dwell_times_s, bins=30, color="#2563eb", edgecolor="#0f172a", linewidth=.45)
        axis.set(xlabel="F=1 dwell time [us]", ylabel="Visits", title="Repumped F=1 dwell times")
        axis.grid(alpha=.22)
        output_files.append(_save(figure, output_directory / "f1_dwell_time_histogram.png"))

        sorted_us = np.sort(1e6 * dwell_times_s)
        survival = 1.0 - np.arange(len(sorted_us), dtype=float) / len(sorted_us)
        figure, axis = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        axis.step(sorted_us, survival, where="post", color="#0f766e")
        axis.set(xlabel="F=1 dwell time [us]", ylabel="Survival fraction", title="F=1 dwell-time survival curve")
        axis.grid(alpha=.22)
        output_files.append(_save(figure, output_directory / "f1_dwell_time_survival_curve.png"))

    if len(photons_per_return):
        figure, axis = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        bins = np.arange(0.5, max(photons_per_return) + 1.5, 1.0)
        axis.hist(photons_per_return, bins=bins, color="#7c3aed", edgecolor="#0f172a", linewidth=.45)
        axis.set(xlabel="Repump absorptions before return to F=2", ylabel="Returns", title="Repump photons per F=1 return")
        axis.grid(alpha=.22)
        output_files.append(_save(figure, output_directory / "repump_photons_per_f1_return.png"))

    if repump_events:
        figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
        labels = list(beam_counts.keys())
        axis.bar(range(len(labels)), [beam_counts[label] for label in labels], color="#c2410c")
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.set(ylabel="Repump absorptions", title="Repump event count by beam")
        axis.grid(alpha=.2, axis="y")
        output_files.append(_save(figure, output_directory / "repump_event_count_by_beam.png"))

        figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
        labels = list(channel_counts.keys())
        axis.bar(range(len(labels)), [channel_counts[label] for label in labels], color="#0891b2")
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set(ylabel="Repump absorptions", title="Repump event count by channel")
        axis.grid(alpha=.2, axis="y")
        output_files.append(_save(figure, output_directory / "repump_event_count_by_channel.png"))

    figure, axis = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    axis.bar(["F=1", "not F=1"], time_fractions, color=("#111827", "#2563eb"))
    axis.set(ylabel="Fraction of simulated time", ylim=(0.0, 1.0), title="Time fraction by ground manifold")
    axis.grid(alpha=.2, axis="y")
    output_files.append(_save(figure, output_directory / "time_fraction_f1_vs_not_f1.png"))
    return output_files


def run_diagnostics(root: Path | None = None, seed: int = 20260817, ensemble_size: int = 10) -> dict[str, object]:
    from .screening import animate_hyperfine_walk
    from .screening import lifetime_file_tag
    from .screening import plot_velocity_and_beam_rates

    paths = multilevel_mot_paths(root); structure = build_atomic_structure(); beams = build_multilevel_cooling_beams()
    coil = default_anti_helmholtz_config()
    config = replace(default_multilevel_mot_config(), dark_state_behavior=DarkStateBehavior.BALLISTIC)
    figures = paths["figures"]; trajectories = paths["trajectories"]; statistics = paths["statistics"]
    output_files = [plot_force_curves(structure, beams, coil, config, figures / "force_curves_unpolarized_f2.png"), plot_field_gradients(coil, figures / "magnetic_field_component_gradients.png")]
    launches = (((4e-3, 1e-3, 2e-3), (-3.0, -0.5, -1.0), 0), ((-3e-3, 2e-3, 1e-3), (2.0, -1.0, .5), 1), ((1e-3, -2e-3, 5e-3), (.5, 1.0, -4.0), -1))
    run_summaries = []
    for index, (position, velocity, m_f) in enumerate(launches):
        initial = MultilevelAtomState(position, velocity, structure.state_index("ground", 2, m_f))
        record = simulate_multilevel_trajectory(initial, 200e-6, coil, beams=beams, structure=structure, config=config, seed=seed + index, max_events=5_000)
        output_files += [
            plot_trajectory(record, structure, trajectories / f"trajectory_{index:02d}.png", f"Multilevel single-atom demonstration {index}"),
            plot_velocity_and_beam_rates(record, beams, trajectories / f"trajectory_{index:02d}_velocity_and_beam_rates.png", index),
            plot_internal_state_timeline(record, structure, trajectories / f"internal_state_timeline_{index:02d}.png", f"Internal-state history: trajectory {index}"),
            animate_hyperfine_walk(record, structure, trajectories / f"hyperfine_walk_trajectory_{index:02d}_{lifetime_file_tag(record)}.gif"),
            _save_trajectory_csv(record, structure, trajectories / f"trajectory_{index:02d}.csv"),
        ]
        run_summaries.append({"index": index, "initial_mF": m_f, "termination": record.termination_reason, **asdict(record.counters)})
    rng = np.random.default_rng(seed + 1000); photon_counts, dark_times_us, parents = [], [], []
    for index in range(ensemble_size):
        m_f = int(rng.integers(-2, 3)); initial = MultilevelAtomState((0, 0, 0), (0, 0, 0), structure.state_index("ground", 2, m_f))
        record = simulate_multilevel_trajectory(initial, 1e-3, coil, beams=beams, structure=structure, config=config, seed=seed + 10_000 + index, max_events=3_000)
        if record.counters.dark_entry_time_s is not None:
            photon_counts.append(record.counters.photons_before_dark); dark_times_us.append(1e6 * record.counters.dark_entry_time_s); parents.append(record.counters.dark_parent_excited_f)
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    panels[0].hist(photon_counts, bins=25, color="#0f766e"); panels[0].set(xlabel="Spontaneous photons before F=1", ylabel="Atoms", title="Dark-state photon budget")
    panels[1].hist(dark_times_us, bins=25, color="#7c3aed"); panels[1].set(xlabel="Dark-entry time [µs]", ylabel="Atoms", title="Optical-pumping time")
    for panel in panels: panel.grid(alpha=.2)
    output_files.append(_save(figure, statistics / "dark_state_ensemble.png"))
    summary = {"seed": seed, "ensemble_size": ensemble_size, "dark_count": len(photon_counts), "dark_fraction": len(photon_counts) / ensemble_size, "mean_photons_before_dark": float(np.mean(photon_counts)) if photon_counts else None, "median_dark_entry_time_us": float(np.median(dark_times_us)) if dark_times_us else None, "dark_parent_excited_F_counts": {str(f): parents.count(f) for f in sorted(set(parents))}, "single_runs": run_summaries, "outputs": [str(path) for path in output_files], "interpretation": "Force curves are instantaneous averages over equal F=2 mF populations. Trajectories use stochastic internal-state events; after entering uncoupled F=1, diagnostic histories continue ballistically to the requested elapsed time."}
    statistics.mkdir(parents=True, exist_ok=True); summary_path = statistics / "diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n"); summary["outputs"].append(str(summary_path))
    return summary


def run_repump_diagnostics(root: Path | None = None, seed: int = 20260819, trajectory_count: int = 8) -> dict[str, object]:
    """Generate focused first-pass repumper diagnostic data and plots."""

    paths = multilevel_mot_paths(root)
    structure = build_atomic_structure()
    config = replace(
        default_multilevel_mot_config(),
        repumper_enabled=True,
        dark_state_behavior=DarkStateBehavior.BALLISTIC,
    )
    beams = build_multilevel_mot_beams(config=config)
    coil = default_anti_helmholtz_config()
    output = paths["trajectories"] / "repump_diagnostics"
    figures = paths["figures"] / "repump_diagnostics"
    rng = np.random.default_rng(seed)
    records: list[MultilevelTrajectoryRecord] = []
    output_files: list[Path] = []
    for index in range(trajectory_count):
        initial_m_f = int(rng.integers(-1, 2))
        initial = MultilevelAtomState((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), structure.state_index("ground", 1, initial_m_f))
        record = simulate_multilevel_trajectory(initial, 200e-6, coil, beams=beams, structure=structure, config=config, seed=seed + index, max_events=20_000)
        records.append(record)
        output_files += [
            _save_trajectory_csv(record, structure, output / f"repumped_trajectory_{index:02d}.csv"),
            _save_repump_absorption_csv(record, output / f"repumped_trajectory_{index:02d}_repump_absorptions.csv"),
            _save_repump_diagnostics(record, output / f"repumped_trajectory_{index:02d}_repump_diagnostics.npz"),
        ]
    output_files += plot_repump_diagnostics(records, structure, beams, figures)
    dwell_times = [duration for record in records for duration in record.f1_visit_durations_s]
    summary = {
        "seed": seed,
        "trajectory_count": trajectory_count,
        "repumper_enabled": True,
        "repump_power_mw_per_beam": 1e3 * config.repump_power_w_per_beam,
        "repump_detuning_mhz": config.repump_detuning_rad_per_s / (2.0 * pi * 1e6),
        "enabled_repump_excited_manifolds": list(config.enabled_repump_excited_manifolds),
        "f1_visit_count": sum(record.counters.f1_visit_count for record in records),
        "repump_absorption_events": sum(record.counters.repump_absorption_events for record in records),
        "f1_returns_to_f2": sum(record.counters.f1_returns_to_f2 for record in records),
        "mean_f1_dwell_time_us": float(1e6 * np.mean(dwell_times)) if dwell_times else None,
        "median_f1_dwell_time_us": float(1e6 * np.median(dwell_times)) if dwell_times else None,
        "outputs": [str(path) for path in output_files],
    }
    summary_path = output / "repump_diagnostic_summary.json"
    output.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["outputs"].append(str(summary_path))
    return summary


def main() -> int:
    print(json.dumps(run_diagnostics(), indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
