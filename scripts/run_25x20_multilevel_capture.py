"""Checkpointed 25-disc x 20-point Section-12 capture/loading campaign.

Every launch velocity on one disc is parallel to its incident normal. The
scalar capture-boundary spectrum is provisional until velocity-mask and
integration convergence checks pass; unresolved rays never count as escaped.
"""

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
from scipy.stats import t as student_t

from pmot.capture_statistics import CaptureVelocitySample, save_capture_spectrum
from pmot.loading import calculate_loading_rate_from_spectrum, save_loading_rate_result
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    CaptureSearchConfig,
    IndeterminateCaptureError,
    build_rate_equation_model,
    capture_cross_section_spectrum,
    default_multilevel_mot_config,
    find_capture_velocity,
    generate_capture_launches,
    multilevel_mot_paths,
)


RUN_NAME = "capture_loading_25x20_r15mm_27mW_dt5us_seed20260921"
SEARCH = CaptureSearchConfig(
    radial_distance_m=15.0e-3,
    disc_radius_m=15.0e-3,
    disc_count=25,
    points_per_disc=20,
    include_center_point=False,
    initial_velocity_guess_m_per_s=30.0,
    velocity_tolerance_m_per_s=0.5,
    maximum_bracket_speed_m_per_s=80.0,
    maximum_simulation_time_s=20.0e-3,
    analysis_velocity_min_m_per_s=0.0,
    analysis_velocity_max_m_per_s=60.0,
    analysis_velocity_step_m_per_s=0.5,
    analysis_s_bin_count=20,
    seed=20260921,
    worker_count=1,
)
TIMEOUT_LADDER_S = (20.0e-3, 40.0e-3, 80.0e-3, 160.0e-3)
TRAPPED_REASONS = {"two_core_entries", "bounded_core_residence"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _worker(payload):
    point, config, coil = payload
    started = perf_counter()
    failures = []
    for timeout_s in TIMEOUT_LADDER_S:
        search = replace(SEARCH, maximum_simulation_time_s=timeout_s)
        try:
            sample = find_capture_velocity(point, search, config=config, coil_config=coil)
        except IndeterminateCaptureError as error:
            failures.append({"timeout_s": timeout_s, "message": str(error)})
            continue
        return {
            "status": "resolved",
            "sample": asdict(sample),
            "maximum_simulation_time_s": timeout_s,
            "prior_indeterminate_attempts": failures,
            "worker_wall_time_s": perf_counter() - started,
        }
    return {
        "status": "indeterminate",
        "disc_index": point.disc_index,
        "point_index": point.point_index,
        "failures": failures,
        "worker_wall_time_s": perf_counter() - started,
    }


def _check_geometry(discs, points) -> None:
    if len(discs) != 25 or len(points) != 500:
        raise RuntimeError("launch geometry is not 25 direction discs x 20 points")
    for disc in discs:
        subset = [point for point in points if point.disc_index == disc.disc_index]
        if len(subset) != 20:
            raise RuntimeError("a direction disc does not have 20 points")
        positions = set()
        for point in subset:
            direction = np.asarray(point.incident_unit_vector)
            normal = np.asarray(disc.incident_unit_vector)
            displacement = np.asarray(point.initial_position_m) - np.asarray(disc.center_position_m)
            if not np.allclose(direction, normal, rtol=0.0, atol=1.0e-14):
                raise RuntimeError("launch velocities on a disc are not parallel")
            if abs(np.dot(displacement, normal)) > 1.0e-12:
                raise RuntimeError("a sampled position is outside its incident disc plane")
            if np.linalg.norm(displacement) > SEARCH.disc_radius_m + 1.0e-12:
                raise RuntimeError("a sampled point lies outside the disc")
            positions.add(point.initial_position_m)
        if len(positions) != 20:
            raise RuntimeError("sampled points on a disc are not distinct")


def _save_geometry(points, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["disc_index", "point_index", "impact_parameter_m", "x0_m", "y0_m", "z0_m", "vx_hat", "vy_hat", "vz_hat"]
        )
        for point in points:
            writer.writerow(
                [point.disc_index, point.point_index, point.s_m,
                 *point.initial_position_m, *point.incident_unit_vector]
            )


def _analyze(samples: list[CaptureVelocitySample], statistics: Path, figures: Path) -> None:
    spectrum = capture_cross_section_spectrum(samples, SEARCH)
    save_capture_spectrum(spectrum, statistics)
    velocity = np.asarray([item.velocity_m_per_s for item in spectrum])
    sigma = np.asarray([item.capture_cross_section_m2 for item in spectrum])
    disc_area = np.pi * SEARCH.disc_radius_m**2
    disc_spectra = []
    disc_loading = []
    for disc_index in range(SEARCH.disc_count):
        group = [sample for sample in samples if sample.disc_index == disc_index]
        if len(group) != SEARCH.points_per_disc:
            raise RuntimeError("missing launch result in a direction disc")
        mask = np.asarray(
            [sample.lower_classification in TRAPPED_REASONS for sample in group]
        )
        thresholds = np.asarray([sample.capture_velocity_m_per_s for sample in group])
        per_disc_sigma = disc_area * np.mean(
            mask[None, :] & (thresholds[None, :] >= velocity[:, None] - 1.0e-12),
            axis=1,
        )
        disc_spectra.append(per_disc_sigma)
        disc_loading.append(calculate_loading_rate_from_spectrum(velocity, per_disc_sigma).loading_rate_atoms_per_s)
    disc_spectra = np.asarray(disc_spectra)
    disc_loading = np.asarray(disc_loading)
    np.testing.assert_allclose(np.mean(disc_spectra, axis=0), sigma, rtol=0.0, atol=1.0e-18)
    t_critical = float(student_t.ppf(0.975, SEARCH.disc_count - 1))
    sem = np.std(disc_spectra, axis=0, ddof=1) / np.sqrt(SEARCH.disc_count)
    sigma_low = np.maximum(0.0, sigma - t_critical * sem)
    sigma_high = sigma + t_critical * sem
    with (statistics / "capture_cross_section_with_95pct_disc_intervals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["velocity_m_per_s", "capture_cross_section_m2", "lower_95pct_m2", "upper_95pct_m2", "captured_count", "launched_count"]
        )
        for v, central, low, high, item in zip(velocity, sigma, sigma_low, sigma_high, spectrum):
            writer.writerow([v, central, low, high, item.captured_count, item.launched_count])
    figure, axis = plt.subplots(figsize=(9, 5.7), constrained_layout=True)
    axis.plot(velocity, 1.0e6 * sigma, color="#0f766e", linewidth=2.0, label="25-disc mean")
    axis.fill_between(
        velocity, 1.0e6 * sigma_low, 1.0e6 * sigma_high,
        color="#99c8c2", alpha=0.55, label="95% Student-t interval across discs",
    )
    axis.set(
        xlabel="Incident speed [m/s]", ylabel="Capture cross section [mm²]",
        title="24-state MOT capture cross section · 25 full-sphere discs × 20 points · 27 mW/beam",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(figures / "capture_cross_section_vs_velocity.png", dpi=180)
    plt.close(figure)

    loading = calculate_loading_rate_from_spectrum(velocity, sigma)
    # Same fitted capture boundaries, sampled on a finer velocity grid, to
    # separate quadrature error from trajectory/threshold uncertainty.
    fine_velocity = np.arange(
        SEARCH.analysis_velocity_min_m_per_s,
        SEARCH.analysis_velocity_max_m_per_s + 0.05,
        0.1,
    )
    all_thresholds = np.asarray([sample.capture_velocity_m_per_s for sample in samples])
    all_valid = np.asarray([sample.lower_classification in TRAPPED_REASONS for sample in samples])
    fine_sigma = disc_area * np.mean(
        all_valid[None, :] & (all_thresholds[None, :] >= fine_velocity[:, None] - 1.0e-12),
        axis=1,
    )
    fine_loading = calculate_loading_rate_from_spectrum(fine_velocity, fine_sigma)
    save_loading_rate_result(loading, statistics / "loading_rate_result.json")
    loading_sem = float(np.std(disc_loading, ddof=1) / np.sqrt(SEARCH.disc_count))
    loading_interval = (
        max(0.0, loading.loading_rate_atoms_per_s - t_critical * loading_sem),
        loading.loading_rate_atoms_per_s + t_critical * loading_sem,
    )
    summary = {
        "schema": "pmot.mot_multilevel.capture-loading-25x20.v1",
        "sample_count": len(samples),
        "independent_direction_discs": SEARCH.disc_count,
        "points_per_disc": SEARCH.points_per_disc,
        "disc_radius_m": SEARCH.disc_radius_m,
        "mean_capture_speed_m_per_s": float(np.mean([sample.capture_velocity_m_per_s for sample in samples])),
        "loading_rate_atoms_per_s": loading.loading_rate_atoms_per_s,
        "loading_rate_0p1mps_quadrature_check_atoms_per_s": fine_loading.loading_rate_atoms_per_s,
        "loading_quadrature_relative_difference": abs(
            fine_loading.loading_rate_atoms_per_s - loading.loading_rate_atoms_per_s
        ) / fine_loading.loading_rate_atoms_per_s,
        "loading_rate_95pct_disc_interval_atoms_per_s": loading_interval,
        "t_degrees_of_freedom": SEARCH.disc_count - 1,
        "warning": "scalar capture boundaries assume local monotonicity; velocity-mask and timestep/duration convergence are not yet established",
    }
    _write_json(statistics / "campaign_summary.json", summary)
    print(f"Loading rate: {loading.loading_rate_atoms_per_s:.6g} atoms/s; 95% disc interval {loading_interval}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--indices", type=int, nargs="*", help="Pilot only: chosen zero-based launch indices")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not np.isclose(SEARCH.time_step_s, 5.0e-6, rtol=0.0, atol=1.0e-18):
        raise RuntimeError("the dt5us campaign must use the 5 microsecond default")
    config = default_multilevel_mot_config()
    if not np.isclose(config.cooling_power_w_per_beam, 27.0e-3, rtol=0.0, atol=1.0e-15):
        raise RuntimeError("the requested 27 mW/beam configuration is not active")
    coil = default_anti_helmholtz_config()
    discs, points = generate_capture_launches(SEARCH)
    _check_geometry(discs, points)
    paths = multilevel_mot_paths()
    statistics = paths["statistics"] / RUN_NAME
    figures = paths["figures"] / RUN_NAME
    result_directory = statistics / "rays"
    manifest_path = statistics / "capture_run_config.json"
    if statistics.exists() and not manifest_path.exists() and any(statistics.iterdir()):
        raise FileExistsError("existing output has no matching campaign manifest")
    statistics.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "pmot.mot_multilevel.capture-25x20-config.v1",
        "search": asdict(SEARCH), "timeout_ladder_s": TIMEOUT_LADDER_S,
        "mot": asdict(config), "coil": asdict(coil),
        "geometry": "25 independent full-sphere direction discs; 20 uniform-area points on each perpendicular disc; all velocities on a disc parallel to its common inward normal",
    }
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("saved campaign configuration does not match")
    else:
        _write_json(manifest_path, manifest)
    geometry_path = statistics / "launch_geometry.csv"
    if not geometry_path.exists():
        _save_geometry(points, geometry_path)

    chosen = range(len(points)) if args.indices is None else sorted(set(args.indices))
    chosen = list(chosen)
    if not chosen or any(index < 0 or index >= len(points) for index in chosen):
        raise ValueError("selected launch indices must be in 0..499")
    pending = []
    for index in chosen:
        point = points[index]
        result_path = result_directory / f"disc_{point.disc_index:02d}_point_{point.point_index:02d}.json"
        if not result_path.exists() or json.loads(result_path.read_text(encoding="utf-8"))["status"] != "resolved":
            pending.append(point)
    print(f"Geometry verified: 25 discs × 20 parallel-velocity launches. Pending {len(pending)} of {len(chosen)} selected rays.", flush=True)
    started = perf_counter()
    # ARC may create/update its SQLite cache during its first atomic-data
    # construction. Preload once before forking so workers inherit the cached
    # 24-state model and never race while creating the same cache index.
    build_rate_equation_model()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_worker, (point, config, coil)): point for point in pending
        }
        for future in as_completed(futures):
            point = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {
                    "status": "error", "disc_index": point.disc_index,
                    "point_index": point.point_index, "error": repr(error),
                }
            result_path = result_directory / f"disc_{point.disc_index:02d}_point_{point.point_index:02d}.json"
            _write_json(result_path, result)
            print(
                f"[{len(chosen) - len(pending) + sum(f.done() for f in futures)}/{len(chosen)}] "
                f"disc={point.disc_index:02d} point={point.point_index:02d} "
                f"status={result['status']} "
                f"vc={result.get('sample', {}).get('capture_velocity_m_per_s', 'n/a')} "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )
    if args.indices is not None:
        print("Pilot complete; full 500-ray analysis not attempted with --indices.", flush=True)
        return
    results = [
        json.loads((result_directory / f"disc_{point.disc_index:02d}_point_{point.point_index:02d}.json").read_text(encoding="utf-8"))
        for point in points
    ]
    unresolved = [index for index, result in enumerate(results) if result["status"] != "resolved"]
    if unresolved:
        _write_json(
            statistics / "unresolved_rays.json",
            {"count": len(unresolved), "indices": unresolved, "results": [results[index] for index in unresolved]},
        )
        raise RuntimeError(f"{len(unresolved)} rays remain unresolved; no full cross section or loading rate was computed")
    samples = [CaptureVelocitySample(**result["sample"]) for result in results]
    _analyze(samples, statistics, figures)


if __name__ == "__main__":
    main()
