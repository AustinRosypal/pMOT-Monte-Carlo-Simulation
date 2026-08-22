"""Ensemble temperature diagnostic for the efficient multilevel MOT."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from ..configuration import HBAR_J_S, RB87_MASS_KG
from ..mot.magnetic_fields import default_anti_helmholtz_config
from .configuration import default_multilevel_mot_config, multilevel_mot_paths
from .rate_equations import (
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    build_rate_equation_model,
    simulate_rate_equation_trajectory,
)
from .simulation import build_multilevel_mot_beams


BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23


def doppler_temperature_k(linewidth_rad_per_s: float) -> float:
    """Return T_D = hbar Gamma/(2 k_B) for angular linewidth Gamma."""

    if linewidth_rad_per_s <= 0.0:
        raise ValueError("linewidth_rad_per_s must be positive")
    return HBAR_J_S * linewidth_rad_per_s / (2.0 * BOLTZMANN_CONSTANT_J_PER_K)


def temperature_components_k(velocities_m_per_s: np.ndarray) -> np.ndarray:
    """Return Tx, Ty, Tz after subtracting ensemble center-of-mass motion."""

    velocities = np.asarray(velocities_m_per_s, dtype=float)
    if velocities.ndim != 2 or velocities.shape[1] != 3 or len(velocities) < 2:
        raise ValueError("velocities must have shape (N, 3) with N >= 2")
    centered = velocities - np.mean(velocities, axis=0, keepdims=True)
    return RB87_MASS_KG * np.mean(centered**2, axis=0) / BOLTZMANN_CONSTANT_J_PER_K


def _temperature_worker(payload):
    atom_index, position, velocity, seed, duration_s, time_step_s, stride = payload
    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    model = build_rate_equation_model(config.natural_linewidth_rad_per_s)
    beams = build_multilevel_mot_beams(config=config)
    record = simulate_rate_equation_trajectory(
        RateEquationAtomState(tuple(position), tuple(velocity)),
        duration_s,
        default_anti_helmholtz_config(),
        beams=beams,
        model=model,
        config=config,
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=time_step_s,
            include_diffusion=True,
            seed=seed,
            escape_radius_m=30.0e-3,
        ),
    )
    times = np.asarray(record.times_s)
    positions = np.asarray(record.positions_m)
    velocities = np.asarray(record.velocities_m_per_s)
    complete = record.termination_reason == "duration" and times[-1] >= duration_s - 1e-12
    final_window = positions[times >= max(0.0, times[-1] - 2.0e-3)]
    trapped = bool(complete and np.max(np.linalg.norm(final_window, axis=1)) <= 2.0e-3)
    indices = np.arange(0, len(times), stride)
    if indices[-1] != len(times) - 1:
        indices = np.append(indices, len(times) - 1)
    return atom_index, trapped, record.termination_reason, times[indices], positions[indices], velocities[indices]


def run_trapped_temperature_study(
    *,
    atom_count: int = 128,
    duration_s: float = 25.0e-3,
    time_step_s: float = 5.0e-6,
    initial_temperature_k: float = 2.0e-3,
    initial_position_sigma_m: float = 0.25e-3,
    seed: int = 20260821,
    worker_count: int = 8,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
) -> dict[str, object]:
    """Run a recoil-diffusing trapped ensemble and measure its late temperature."""

    if atom_count < 2 or duration_s <= 0.0 or time_step_s <= 0.0 or worker_count <= 0:
        raise ValueError("atom_count >= 2 and positive duration, timestep, and workers are required")
    paths = multilevel_mot_paths()
    output = output_directory or paths["statistics"] / "trapped_temperature_25ms"
    figures = figure_directory or paths["figures"] / "trapped_temperature_25ms"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    positions = rng.normal(scale=initial_position_sigma_m, size=(atom_count, 3))
    velocity_sigma = np.sqrt(BOLTZMANN_CONSTANT_J_PER_K * initial_temperature_k / RB87_MASS_KG)
    velocities = rng.normal(scale=velocity_sigma, size=(atom_count, 3))
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=atom_count, dtype=np.uint32)
    stride = max(1, int(round(0.1e-3 / time_step_s)))
    payloads = [
        (index, positions[index], velocities[index], int(seeds[index]), duration_s, time_step_s, stride)
        for index in range(atom_count)
    ]

    started = perf_counter()
    completed = 0
    records = []
    print(f"[temperature] simulation 0/{atom_count}; workers={worker_count}", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_temperature_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            records.append(future.result())
            completed += 1
            elapsed = perf_counter() - started
            eta = (atom_count - completed) * elapsed / completed
            print(f"[temperature] simulation {completed}/{atom_count}; ETA={eta/60:.1f} min", flush=True)
    records.sort(key=lambda item: item[0])
    trapped_records = [item for item in records if item[1]]
    if len(trapped_records) < 2:
        raise RuntimeError(f"only {len(trapped_records)} atoms met the final bounded-core criterion")
    reference_times = trapped_records[0][3]
    aligned = [item for item in trapped_records if len(item[3]) == len(reference_times) and np.allclose(item[3], reference_times)]
    if len(aligned) < 2:
        raise RuntimeError("fewer than two trapped records share the complete time grid")
    velocity_history = np.stack([item[5] for item in aligned])
    component_history = np.stack([
        temperature_components_k(velocity_history[:, time_index, :])
        for time_index in range(len(reference_times))
    ])
    mean_history = np.mean(component_history, axis=1)
    plateau_start_s = max(0.0, duration_s - 5.0e-3)
    plateau_mask = reference_times >= plateau_start_s - 1e-12
    plateau_indices = np.flatnonzero(plateau_mask)
    window_groups = np.array_split(plateau_indices, 4)
    window_means = np.asarray([np.mean(mean_history[group]) for group in window_groups])
    stationary_temperature = float(np.mean(mean_history[plateau_mask]))
    stationarity_metric = float((np.max(window_means) - np.min(window_means)) / stationary_temperature)
    final_components = component_history[-1]
    final_temperature = float(np.mean(final_components))
    doppler = doppler_temperature_k(default_multilevel_mot_config().natural_linewidth_rad_per_s)
    initial_components = temperature_components_k(velocities)
    anisotropy = float(np.max(final_components) / np.min(final_components))

    csv_path = output / "temperature_vs_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "temperature_x_k", "temperature_y_k", "temperature_z_k", "temperature_mean_k"))
        writer.writerows(zip(reference_times, component_history[:, 0], component_history[:, 1], component_history[:, 2], mean_history, strict=True))

    figure, axis = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    for component, label, color in zip(component_history.T, ("Tx", "Ty", "Tz"), ("#dc2626", "#2563eb", "#16a34a"), strict=True):
        axis.plot(1e3 * reference_times, 1e6 * component, label=label, color=color, alpha=0.75)
    axis.plot(1e3 * reference_times, 1e6 * mean_history, label="T mean", color="#111827", linewidth=2.0)
    axis.axhline(1e6 * doppler, color="#7c3aed", linestyle="--", label=f"Doppler T = {1e6*doppler:.1f} µK")
    axis.axvspan(1e3 * plateau_start_s, 1e3 * duration_s, color="#fbbf24", alpha=0.12, label="plateau window")
    axis.set(xlabel="time [ms]", ylabel="temperature [µK]", title="Full multilevel MOT trapped-ensemble temperature")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3)
    plot_path = figures / "temperature_vs_time.png"
    figure.savefig(plot_path, dpi=190)
    plt.close(figure)

    config = replace(default_multilevel_mot_config(), repumper_enabled=True)
    payload = {
        "definition": "T_i = m Var(v_i - <v_i>)/k_B; T=(Tx+Ty+Tz)/3",
        "doppler_equation": "T_D = hbar Gamma/(2 k_B)",
        "linewidth_rad_per_s": config.natural_linewidth_rad_per_s,
        "linewidth_hz": config.natural_linewidth_rad_per_s / (2.0 * np.pi),
        "doppler_temperature_k": doppler,
        "requested_atom_count": atom_count,
        "trapped_atom_count": len(aligned),
        "duration_s": duration_s,
        "time_step_s": time_step_s,
        "initial_target_temperature_k": initial_temperature_k,
        "initial_measured_components_k": initial_components.tolist(),
        "initial_measured_temperature_k": float(np.mean(initial_components)),
        "final_temperature_components_k": final_components.tolist(),
        "final_temperature_k": final_temperature,
        "stationary_temperature_k": stationary_temperature,
        "stationary_over_doppler": stationary_temperature / doppler,
        "plateau_window_s": duration_s - plateau_start_s,
        "plateau_four_window_means_k": window_means.tolist(),
        "stationarity_metric": stationarity_metric,
        "stationarity_requirement": 0.15,
        "stationarity_pass": stationarity_metric < 0.15,
        "final_anisotropy_ratio": anisotropy,
        "wall_time_s": perf_counter() - started,
        "seed": seed,
        "multilevel_config": asdict(config),
        "limitations": [
            "This is the MOT configuration, not the clean zero-field low-saturation optical-molasses benchmark.",
            "The population-rate model excludes coherent and sub-Doppler cooling.",
            "A separate dt/2 ensemble is required for formal timestep-convergence validation.",
        ],
        "outputs": {"temperature_csv": str(csv_path), "temperature_plot": str(plot_path)},
    }
    summary_path = output / "temperature_summary.json"
    payload["outputs"]["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure the full-MOT trapped-ensemble temperature")
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--duration-ms", type=float, default=25.0)
    parser.add_argument("--dt-us", type=float, default=5.0)
    parser.add_argument("--initial-temperature-mk", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_trapped_temperature_study(
        atom_count=args.atoms,
        duration_s=1e-3 * args.duration_ms,
        time_step_s=1e-6 * args.dt_us,
        initial_temperature_k=1e-3 * args.initial_temperature_mk,
        worker_count=args.workers,
        seed=args.seed,
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

