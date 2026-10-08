"""Adaptive conventional multilevel-MOT loading versus axial field gradient.

This is a diagnostic Section-12 population-rate campaign.  It uses common
full-sphere launch geometry at every gradient, treats direction discs as the
independent statistical clusters, and extends in 25-disc batches until every
loading-rate point has a 95% Student-t half-width no larger than 10%.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from math import pi
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from pmot.capture_statistics import CaptureVelocitySample
from pmot.loading import calculate_loading_rate_from_spectrum
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    CaptureSearchConfig,
    IndeterminateCaptureError,
    build_rate_equation_model,
    default_multilevel_mot_config,
    find_capture_velocity,
    generate_capture_launches,
)
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams


RUN_NAME = (
    "loading_vs_axial_gradient_10_15_20_25_30Gpcm_"
    "22p76MHz_27mW_repump0p1mW_adaptive_20261004"
)
GRADIENTS_G_PER_CM = (10.0, 15.0, 20.0, 25.0, 30.0)
MINIMUM_DISCS = 100
DISC_BATCH = 25
MAXIMUM_DISCS = 300
POINTS_PER_DISC = 20
RELATIVE_CI95_TARGET = 0.10
TIMEOUT_LADDER_S = (50.0e-3, 100.0e-3, 200.0e-3, 400.0e-3)
TRAPPED_REASONS = {"two_core_entries", "bounded_core_residence"}
SEARCH = CaptureSearchConfig(
    radial_distance_m=15.0e-3,
    disc_radius_m=15.0e-3,
    disc_count=MAXIMUM_DISCS,
    points_per_disc=POINTS_PER_DISC,
    include_center_point=False,
    initial_velocity_guess_m_per_s=30.0,
    velocity_tolerance_m_per_s=0.25,
    maximum_bracket_speed_m_per_s=80.0,
    max_bracket_iterations=16,
    max_search_iterations=16,
    maximum_simulation_time_s=TIMEOUT_LADDER_S[0],
    time_step_s=5.0e-6,
    analysis_velocity_min_m_per_s=0.0,
    analysis_velocity_max_m_per_s=60.0,
    analysis_velocity_step_m_per_s=0.25,
    analysis_s_bin_count=20,
    seed=20261004,
    worker_count=1,
    checkpoint_every=1,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _gradient_label(gradient_g_per_cm: float) -> str:
    return f"{gradient_g_per_cm:g}Gpcm".replace(".", "p")


def _result_path(root: Path, gradient_g_per_cm: float, point) -> Path:
    return (
        root
        / "rays"
        / _gradient_label(gradient_g_per_cm)
        / f"disc_{point.disc_index:03d}_point_{point.point_index:02d}.json"
    )


def _mot_config():
    return replace(
        default_multilevel_mot_config(),
        cooling_detuning_rad_per_s=-2.0 * pi * 22.76e6,
        repump_detuning_rad_per_s=0.0,
        cooling_power_w_per_beam=27.0e-3,
        repump_power_w_per_beam=0.1e-3,
        repumper_enabled=True,
        include_gravity=True,
    )


def _worker(payload):
    gradient_g_per_cm, point, config = payload
    coil = default_anti_helmholtz_config(
        target_gradient_g_per_cm=gradient_g_per_cm
    )
    started = perf_counter()
    failures = []
    for timeout_s in TIMEOUT_LADDER_S:
        search = replace(SEARCH, maximum_simulation_time_s=timeout_s)
        try:
            sample = find_capture_velocity(
                point,
                search,
                config=config,
                coil_config=coil,
            )
        except IndeterminateCaptureError as error:
            failures.append({"timeout_s": timeout_s, "message": str(error)})
            continue
        return {
            "status": "resolved",
            "gradient_g_per_cm": gradient_g_per_cm,
            "sample": asdict(sample),
            "maximum_simulation_time_s": timeout_s,
            "prior_indeterminate_attempts": failures,
            "worker_wall_time_s": perf_counter() - started,
        }
    return {
        "status": "indeterminate",
        "gradient_g_per_cm": gradient_g_per_cm,
        "disc_index": point.disc_index,
        "point_index": point.point_index,
        "failures": failures,
        "worker_wall_time_s": perf_counter() - started,
    }


def _save_geometry(points, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "disc_index",
                "point_index",
                "impact_parameter_m",
                "x0_m",
                "y0_m",
                "z0_m",
                "vx_hat",
                "vy_hat",
                "vz_hat",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.disc_index,
                    point.point_index,
                    point.s_m,
                    *point.initial_position_m,
                    *point.incident_unit_vector,
                ]
            )


def _preflight(root: Path, points) -> None:
    config = _mot_config()
    beams = build_multilevel_mot_beams(config=config)
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    if len(cooling) != 6 or len(repump) != 6:
        raise RuntimeError("expected six cooling and six repump traveling components")
    if not all(np.isclose(beam.power_w, 27.0e-3) for beam in cooling):
        raise RuntimeError("27 mW cooling power is not active")
    if not all(np.isclose(beam.detuning_hz, -22.76e6) for beam in cooling):
        raise RuntimeError("-22.76 MHz cooling detuning is not active")
    if not all(np.isclose(2.0 * beam.beam_radius_m, 12.7e-3) for beam in beams):
        raise RuntimeError("12.7 mm beam diameter is not active")
    if not all(np.isclose(beam.power_w, 0.1e-3) for beam in repump):
        raise RuntimeError("0.1 mW repump power is not active")
    if not all(np.isclose(beam.detuning_hz, 0.0) for beam in repump):
        raise RuntimeError("resonant repump setting is not active")
    coils = {
        str(gradient): asdict(
            default_anti_helmholtz_config(target_gradient_g_per_cm=gradient)
        )
        for gradient in GRADIENTS_G_PER_CM
    }
    manifest = {
        "schema": "pmot.mot_multilevel.gradient-loading-adaptive.v1",
        "scientific_status": (
            "diagnostic Section-12 deterministic mean-force multilevel MOT; "
            "quantitative capture/loading validation remains pending"
        ),
        "run_name": RUN_NAME,
        "gradients_g_per_cm": GRADIENTS_G_PER_CM,
        "mot_config": asdict(config),
        "beam_diameter_m": 12.7e-3,
        "search": asdict(SEARCH),
        "timeout_ladder_s": TIMEOUT_LADDER_S,
        "minimum_discs": MINIMUM_DISCS,
        "disc_batch": DISC_BATCH,
        "maximum_discs_safety_cap": MAXIMUM_DISCS,
        "relative_ci95_target": RELATIVE_CI95_TARGET,
        "coils": coils,
        "statistics": (
            "direction discs are independent clusters; common seeded full-sphere "
            "geometry is reused at every gradient"
        ),
    }
    manifest = json.loads(json.dumps(manifest))
    manifest_path = root / "campaign_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("saved campaign manifest does not match requested inputs")
    else:
        _write_json(manifest_path, manifest)
    geometry_path = root / "launch_geometry.csv"
    if not geometry_path.exists():
        _save_geometry(points, geometry_path)


def _load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_tasks(root: Path, tasks, workers: int) -> None:
    if not tasks:
        return
    config = _mot_config()
    build_rate_equation_model()
    started = perf_counter()
    completed = 0
    recent_report = started
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_worker, (gradient, point, config)): (gradient, point)
            for gradient, point in tasks
        }
        for future in as_completed(futures):
            gradient, point = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "status": "error",
                    "gradient_g_per_cm": gradient,
                    "disc_index": point.disc_index,
                    "point_index": point.point_index,
                    "error": repr(error),
                }
            _write_json(_result_path(root, gradient, point), result)
            completed += 1
            now = perf_counter()
            if now - recent_report >= 45.0 or completed == len(tasks):
                rate = completed / max(now - started, 1.0e-12)
                eta_s = (len(tasks) - completed) / rate if rate > 0.0 else float("inf")
                print(
                    f"[gradient loading] {completed}/{len(tasks)} new rays; "
                    f"Gz={gradient:g} G/cm disc={point.disc_index} "
                    f"point={point.point_index}; status={result['status']}; "
                    f"elapsed={now-started:.1f}s; batch ETA={eta_s/3600:.2f}h",
                    flush=True,
                )
                recent_report = now


def _samples_for_gradient(root: Path, gradient: float, disc_count: int):
    samples = []
    unresolved = []
    for disc in range(disc_count):
        for point in range(POINTS_PER_DISC):
            path = (
                root
                / "rays"
                / _gradient_label(gradient)
                / f"disc_{disc:03d}_point_{point:02d}.json"
            )
            result = _load_result(path)
            if result is None or result.get("status") != "resolved":
                unresolved.append(
                    {
                        "gradient_g_per_cm": gradient,
                        "disc_index": disc,
                        "point_index": point,
                        "result": result,
                    }
                )
                continue
            samples.append(CaptureVelocitySample(**result["sample"]))
    return samples, unresolved


def _analyze(root: Path, disc_count: int) -> dict:
    analysis_root = root / "analysis"
    figures = root / "figures"
    analysis_root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    velocity = np.arange(
        SEARCH.analysis_velocity_min_m_per_s,
        SEARCH.analysis_velocity_max_m_per_s
        + 0.5 * SEARCH.analysis_velocity_step_m_per_s,
        SEARCH.analysis_velocity_step_m_per_s,
    )
    disc_area = pi * SEARCH.disc_radius_m**2
    rows = []
    unresolved_all = []
    combined_cross_sections = {}
    for gradient in GRADIENTS_G_PER_CM:
        samples, unresolved = _samples_for_gradient(root, gradient, disc_count)
        unresolved_all.extend(unresolved)
        if unresolved:
            continue
        disc_spectra = []
        disc_rates = []
        for disc in range(disc_count):
            group = [sample for sample in samples if sample.disc_index == disc]
            if len(group) != POINTS_PER_DISC:
                raise RuntimeError("a complete direction disc is missing launch points")
            valid = np.asarray(
                [sample.lower_classification in TRAPPED_REASONS for sample in group]
            )
            thresholds = np.asarray(
                [sample.capture_velocity_m_per_s for sample in group]
            )
            sigma = disc_area * np.mean(
                valid[None, :]
                & (thresholds[None, :] >= velocity[:, None] - 1.0e-12),
                axis=1,
            )
            disc_spectra.append(sigma)
            disc_rates.append(
                calculate_loading_rate_from_spectrum(
                    velocity, sigma
                ).loading_rate_atoms_per_s
            )
        disc_spectra = np.asarray(disc_spectra)
        disc_rates = np.asarray(disc_rates)
        mean_spectrum = np.mean(disc_spectra, axis=0)
        mean_rate = float(np.mean(disc_rates))
        aggregate_rate = calculate_loading_rate_from_spectrum(
            velocity, mean_spectrum
        ).loading_rate_atoms_per_s
        np.testing.assert_allclose(mean_rate, aggregate_rate, rtol=1.0e-12)
        critical = float(student_t.ppf(0.975, disc_count - 1))
        rate_sem = float(np.std(disc_rates, ddof=1) / np.sqrt(disc_count))
        rate_half = critical * rate_sem
        spectrum_sem = np.std(disc_spectra, axis=0, ddof=1) / np.sqrt(disc_count)
        spectrum_low = np.maximum(0.0, mean_spectrum - critical * spectrum_sem)
        spectrum_high = mean_spectrum + critical * spectrum_sem
        coil = default_anti_helmholtz_config(
            target_gradient_g_per_cm=gradient
        )
        row = {
            "gradient_g_per_cm": gradient,
            "coil_current_a": coil.current_a,
            "independent_direction_discs": disc_count,
            "points_per_disc": POINTS_PER_DISC,
            "resolved_capture_boundaries": len(samples),
            "loading_rate_atoms_per_s": mean_rate,
            "loading_rate_standard_error_atoms_per_s": rate_sem,
            "loading_rate_ci95_low_atoms_per_s": max(0.0, mean_rate - rate_half),
            "loading_rate_ci95_high_atoms_per_s": mean_rate + rate_half,
            "loading_rate_ci95_half_width_atoms_per_s": rate_half,
            "loading_rate_relative_ci95_half_width": (
                rate_half / mean_rate if mean_rate > 0.0 else float("inf")
            ),
            "zero_loading_direction_discs": int(np.count_nonzero(disc_rates == 0.0)),
        }
        rows.append(row)
        combined_cross_sections[gradient] = (
            mean_spectrum,
            spectrum_low,
            spectrum_high,
        )
        gradient_root = analysis_root / _gradient_label(gradient)
        gradient_root.mkdir(parents=True, exist_ok=True)
        with (gradient_root / "loading_rate_by_disc.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["disc_index", "loading_rate_atoms_per_s"])
            writer.writerows(enumerate(disc_rates))
        with (gradient_root / "capture_cross_section_with_95pct_intervals.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "velocity_m_per_s",
                    "mean_cross_section_m2",
                    "ci95_low_m2",
                    "ci95_high_m2",
                ]
            )
            writer.writerows(
                zip(velocity, mean_spectrum, spectrum_low, spectrum_high)
            )
    if unresolved_all:
        _write_json(
            root / "unresolved_rays.json",
            {"count": len(unresolved_all), "rays": unresolved_all},
        )
        raise RuntimeError(
            f"{len(unresolved_all)} rays are unresolved through the timeout ladder"
        )
    aggregate_path = analysis_root / "loading_rate_vs_gradient.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gradients = np.asarray([row["gradient_g_per_cm"] for row in rows])
    currents = np.asarray([row["coil_current_a"] for row in rows])
    rates = np.asarray([row["loading_rate_atoms_per_s"] for row in rows])
    errors = np.asarray([row["loading_rate_ci95_half_width_atoms_per_s"] for row in rows])
    figure, axis = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    axis.errorbar(
        gradients,
        rates,
        yerr=errors,
        fmt="o-",
        color="#0f766e",
        ecolor="#b45309",
        capsize=5,
        linewidth=2.0,
        markersize=7,
        label="95% direction-cluster Student-t interval",
    )
    axis.set(
        xlabel=r"Axial gradient $|\partial B_z/\partial z|$ [G/cm]",
        ylabel="Loading rate [atoms/s]",
        title=(
            "24-state MOT loading versus anti-Helmholtz gradient\n"
            "27 mW cooling at -22.76 MHz; 0.1 mW resonant repump"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    current_per_gradient = float(np.mean(currents / gradients))
    secondary = axis.secondary_xaxis(
        "top",
        functions=(
            lambda gradient: gradient * current_per_gradient,
            lambda current: current / current_per_gradient,
        ),
    )
    secondary.set_xlabel("Coil current for modeled 40 mm, 50-turn pair [A]")
    figure.savefig(figures / "loading_rate_vs_axial_gradient.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 5.9), constrained_layout=True)
    for gradient in GRADIENTS_G_PER_CM:
        mean, low, high = combined_cross_sections[gradient]
        axis.plot(velocity, 1.0e6 * mean, linewidth=1.8, label=f"{gradient:g} G/cm")
        axis.fill_between(velocity, 1.0e6 * low, 1.0e6 * high, alpha=0.10)
    axis.set(
        xlabel="Incident speed [m/s]",
        ylabel=r"Capture cross section [mm$^2$]",
        title=f"Capture cross sections with 95% cluster intervals ({disc_count} discs)",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.savefig(figures / "capture_cross_sections_all_gradients.png", dpi=200)
    plt.close(figure)

    converged = all(
        row["loading_rate_relative_ci95_half_width"] <= RELATIVE_CI95_TARGET
        for row in rows
    )
    summary = {
        "schema": "pmot.mot_multilevel.gradient-loading-adaptive-summary.v1",
        "scientific_status": (
            "diagnostic Section-12 deterministic mean-force multilevel MOT; "
            "not yet a fully validated quantitative loading prediction"
        ),
        "disc_count": disc_count,
        "points_per_disc": POINTS_PER_DISC,
        "relative_ci95_target": RELATIVE_CI95_TARGET,
        "all_gradients_meet_target": converged,
        "gradients": rows,
        "aggregate_csv": str(aggregate_path),
        "loading_plot": str(figures / "loading_rate_vs_axial_gradient.png"),
        "cross_section_plot": str(figures / "capture_cross_sections_all_gradients.png"),
    }
    _write_json(root / "adaptive_summary.json", summary)
    return summary


def _pending_tasks(root: Path, points, disc_count: int):
    tasks = []
    for point in points:
        if point.disc_index >= disc_count:
            continue
        for gradient in GRADIENTS_G_PER_CM:
            result = _load_result(_result_path(root, gradient, point))
            if result is None or result.get("status") != "resolved":
                tasks.append((gradient, point))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="run only disc 0, point 0 at all five gradients",
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    root = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "diagnostics"
        / "mot_multilevel"
        / RUN_NAME
    )
    root.mkdir(parents=True, exist_ok=True)
    _, points = generate_capture_launches(SEARCH)
    _preflight(root, points)
    if args.pilot:
        selected = [point for point in points if point.disc_index == 0 and point.point_index == 0]
        tasks = []
        for point in selected:
            for gradient in GRADIENTS_G_PER_CM:
                result = _load_result(_result_path(root, gradient, point))
                if result is None or result.get("status") != "resolved":
                    tasks.append((gradient, point))
        print(f"Pilot pending: {len(tasks)} of 5 gradient/ray combinations", flush=True)
        _run_tasks(root, tasks, min(args.workers, max(1, len(tasks))))
        return

    target_discs = MINIMUM_DISCS
    while True:
        tasks = _pending_tasks(root, points, target_discs)
        print(
            f"Adaptive target {target_discs} discs: {len(tasks)} unresolved or missing "
            f"gradient/ray combinations",
            flush=True,
        )
        _run_tasks(root, tasks, args.workers)
        summary = _analyze(root, target_discs)
        for row in summary["gradients"]:
            print(
                f"Gz={row['gradient_g_per_cm']:g} G/cm: "
                f"R={row['loading_rate_atoms_per_s']:.6g} atoms/s; "
                f"relative 95% half-width="
                f"{100.0*row['loading_rate_relative_ci95_half_width']:.2f}%",
                flush=True,
            )
        if summary["all_gradients_meet_target"]:
            print(
                f"Adaptive precision target met at {target_discs} direction discs.",
                flush=True,
            )
            return
        target_discs += DISC_BATCH
        if target_discs > MAXIMUM_DISCS:
            raise RuntimeError(
                "precision target was not met before the 300-disc safety cap"
            )


if __name__ == "__main__":
    main()
