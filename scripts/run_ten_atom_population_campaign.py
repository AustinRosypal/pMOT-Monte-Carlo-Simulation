"""Seeded ten-launch diagnostic of the Section-12 24-state MOT.

This is an illustrative, non-converged campaign. Capture boundaries assume
local monotonicity and use the existing early-exit core criterion.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pmot.capture_statistics import CaptureVelocitySample, plot_capture_cross_section, save_capture_spectrum
from pmot.loading import calculate_loading_rate_from_spectrum, save_loading_rate_result
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    CaptureSearchConfig,
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    RateEquationTrajectoryRecord,
    IndeterminateCaptureError,
    build_multilevel_mot_beams,
    build_rate_equation_model,
    capture_cross_section_spectrum,
    classify_capture_trajectory,
    create_population_histogram_animation,
    default_multilevel_mot_config,
    find_capture_velocity,
    generate_capture_launches,
    multilevel_mot_paths,
    plot_time_diagnostics,
    plot_trajectory_3d,
    rate_equation_observable,
    save_trajectory,
    simulate_rate_equation_trajectory,
    trajectory_summary,
)


RUN_NAME = "ten_atom_section12_probe_20260919"
# Preserve the completed probe's 20 mW/beam provenance after the production
# multilevel default changes. A 27 mW campaign needs a new run name/output root.
RUN_COOLING_POWER_W_PER_BEAM = 20.0e-3
SEARCH = CaptureSearchConfig(
    radial_distance_m=15.0e-3,
    disc_radius_m=5.0e-3,
    disc_count=10,
    points_per_disc=1,
    initial_velocity_guess_m_per_s=35.0,
    velocity_tolerance_m_per_s=1.0,
    maximum_simulation_time_s=15.0e-3,
    time_step_s=20.0e-6,
    analysis_velocity_step_m_per_s=1.0,
    analysis_velocity_max_m_per_s=80.0,
    seed=20260919,
)
CAPTURE_TIMEOUT_LADDER_S = (15.0e-3, 30.0e-3, 50.0e-3, 100.0e-3)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_saved_record(path: Path, metadata_path: Path) -> RateEquationTrajectoryRecord:
    with np.load(path) as arrays:
        record = RateEquationTrajectoryRecord(
            times_s=arrays["time_s"].tolist(),
            positions_m=arrays["position_m"].tolist(),
            velocities_m_per_s=arrays["velocity_m_per_s"].tolist(),
            forces_n=arrays["force_n"].tolist(),
            beam_effective_scattering_rates_per_s=arrays[
                "beam_effective_scattering_rate_per_s"
            ].tolist(),
            total_spontaneous_scattering_rates_per_s=arrays[
                "total_spontaneous_scattering_rate_per_s"
            ].tolist(),
            magnetic_fields_t=arrays["magnetic_field_t"].tolist(),
            quantization_axes=arrays["quantization_axis"].tolist(),
            populations=list(arrays["populations"]),
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    record.termination_reason = metadata["summary"]["termination_reason"]
    return record


def _force_validation(model, beams, coil, config, statistics: Path, figures: Path) -> dict:
    """Test zero force at center, three restoring slopes, and field/helicity dependence."""

    optical_config = replace(config, include_gravity=False)
    field_free_coil = replace(coil, current_a=0.0)
    opposite = {"sigma+": "sigma-", "sigma-": "sigma+", "pi": "pi"}
    reversed_beams = [
        replace(beam, circular_polarization=opposite[beam.circular_polarization])
        for beam in beams
    ]
    zero_velocity = (0.0, 0.0, 0.0)

    def force(position, local_beams=beams, local_coil=coil):
        return np.asarray(
            rate_equation_observable(
                model, local_beams, tuple(position), zero_velocity,
                local_coil, optical_config,
            ).force_n,
            dtype=float,
        )

    origin = force(np.zeros(3))
    offsets_m = np.linspace(-1.0e-3, 1.0e-3, 17)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    slopes = []
    for axis_index, axis in enumerate(axes):
        probe = np.zeros(3)
        normal = []
        field_free = []
        reversed_helicity = []
        for offset in offsets_m:
            probe[axis_index] = offset
            normal.append(force(probe)[axis_index])
            field_free.append(force(probe, local_coil=field_free_coil)[axis_index])
            reversed_helicity.append(force(probe, local_beams=reversed_beams)[axis_index])
        normal = np.asarray(normal)
        field_free = np.asarray(field_free)
        reversed_helicity = np.asarray(reversed_helicity)
        axis.plot(1.0e3 * offsets_m, 1.0e21 * normal, label="coils + configured helicities")
        axis.plot(1.0e3 * offsets_m, 1.0e21 * field_free, "--", label="field off")
        axis.plot(1.0e3 * offsets_m, 1.0e21 * reversed_helicity, ":", label="helicities reversed")
        axis.axhline(0.0, color="#64748b", linewidth=0.6)
        axis.axvline(0.0, color="#64748b", linewidth=0.6)
        axis.set(xlabel=f"{('x', 'y', 'z')[axis_index]} displacement [mm]", ylabel=f"F{('x', 'y', 'z')[axis_index]} [zN]")
        axis.grid(alpha=0.25)
        slope = float(np.polyfit(offsets_m[7:10], normal[7:10], 1)[0])
        slopes.append(slope)
        if not slope < 0.0:
            raise RuntimeError(f"nonrestoring optical-force slope on axis {axis_index}")
        if np.max(np.abs(field_free)) > 1.0e-30:
            raise RuntimeError(f"field-free position force is nonzero on axis {axis_index}")
        if not np.allclose(reversed_helicity, -normal, rtol=1.0e-8, atol=1.0e-30):
            raise RuntimeError(f"helicity reversal did not reverse axis {axis_index} force")
    axes[0].legend(fontsize=8)
    figure.suptitle("Magnetic-field/helicity restoring-force check (optical force, gravity excluded)")
    figure_path = figures / "restoring_force_three_axes.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    if np.max(np.abs(origin)) > 1.0e-30:
        raise RuntimeError("symmetric origin optical force is not zero")
    report = {
        "origin_optical_force_n": origin.tolist(),
        "restoring_slopes_n_per_m": slopes,
        "field_free_max_abs_force_n": float(np.max(np.abs(field_free))),
        "beam_polarizations": {beam.label: beam.circular_polarization for beam in beams},
        "gravity_excluded_from_force_check": True,
        "figure": str(figure_path),
    }
    _write_json(statistics / "restoring_force_check.json", report)
    return report


def _combined_plot(rows: list[dict], records: list, path: Path) -> None:
    figure = plt.figure(figsize=(11, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    for row, record in zip(rows, records):
        xyz_mm = 1.0e3 * np.asarray(record.positions_m, dtype=float)
        color = "#0f766e" if row["classification"] == "trapped" else "#b91c1c"
        axis.plot(*xyz_mm.T, color=color, linewidth=1.3, alpha=0.8)
        axis.scatter(*xyz_mm[0], color=color, s=14)
        axis.text(*xyz_mm[-1], str(row["atom"]), color=color, fontsize=8)
    axis.scatter(0, 0, 0, color="#111827", marker="+", s=120, label="trap center")
    axis.set(
        xlim=(-32, 32), ylim=(-32, 32), zlim=(-32, 32),
        xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]",
        title="Ten seeded 24-state MOT trajectories (green: captured, red: escaped)",
    )
    axis.set_box_aspect((1, 1, 1))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    paths = multilevel_mot_paths()
    statistics = paths["statistics"] / RUN_NAME
    figures = paths["figures"] / RUN_NAME
    trajectories = paths["trajectories"] / RUN_NAME
    manifest_path = statistics / "run_config.json"
    if statistics.exists() and not manifest_path.exists() and any(statistics.iterdir()):
        raise FileExistsError("existing statistics without this campaign manifest")
    for directory in (statistics, figures, trajectories):
        directory.mkdir(parents=True, exist_ok=True)
    config = replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=RUN_COOLING_POWER_W_PER_BEAM,
    )
    coil = default_anti_helmholtz_config()
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    manifest = {
        "schema": "pmot.mot_multilevel.ten-atom-probe.v1",
        "warning": "illustrative only; capture monotonicity and timestep/duration convergence not established",
        "search": asdict(SEARCH), "mot": asdict(config), "coil": asdict(coil),
        "speed_selection": "seeded uniform factors [0.60,0.80] and [1.20,1.40] around each local capture boundary",
        "capture_timeout_ladder_s": CAPTURE_TIMEOUT_LADDER_S,
    }
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        former_manifest = {key: value for key, value in manifest.items() if key != "capture_timeout_ladder_s"}
        if saved_manifest == former_manifest:
            _write_json(manifest_path, manifest)
        elif saved_manifest != manifest:
            raise RuntimeError("existing run manifest does not match")
    else:
        _write_json(manifest_path, manifest)
    force_report = _force_validation(model, beams, coil, config, statistics, figures)
    print("Restoring slopes [N/m]:", force_report["restoring_slopes_n_per_m"], flush=True)

    _, points = generate_capture_launches(SEARCH)
    rng = np.random.default_rng(20260920)
    rows: list[dict] = []
    samples: list[CaptureVelocitySample] = []
    records = []
    first_captured_record = None
    for atom_index, point in enumerate(points, start=1):
        factor = rng.uniform(0.60, 0.80) if atom_index % 2 else rng.uniform(1.20, 1.40)
        atom_stem = f"atom_{atom_index:02d}"
        row_path = statistics / f"{atom_stem}_summary.json"
        record_path = trajectories / f"{atom_stem}.npz"
        metadata_path = trajectories / f"{atom_stem}_metadata.json"
        if row_path.exists() and record_path.exists() and metadata_path.exists():
            row = json.loads(row_path.read_text(encoding="utf-8"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            sample = CaptureVelocitySample(**metadata["capture_sample"])
            record = _load_saved_record(record_path, metadata_path)
            row.setdefault("capture_max_time_ms", 1.0e3 * SEARCH.maximum_simulation_time_s)
            rows.append(row)
            samples.append(sample)
            records.append(record)
            if row["classification"] == "trapped" and first_captured_record is None:
                first_captured_record = record
            print(f"Atom {atom_index:02d}: resumed saved {row['classification']}", flush=True)
            continue

        search_for_atom = SEARCH
        for maximum_time_s in CAPTURE_TIMEOUT_LADDER_S:
            search_for_atom = replace(SEARCH, maximum_simulation_time_s=maximum_time_s)
            try:
                sample = find_capture_velocity(
                    point, search_for_atom, coil_config=coil, config=config
                )
                break
            except IndeterminateCaptureError as error:
                print(
                    f"Atom {atom_index:02d}: {maximum_time_s * 1e3:g} ms "
                    f"capture search indeterminate ({error}); extending window",
                    flush=True,
                )
        else:
            raise IndeterminateCaptureError(
                f"atom {atom_index} remained indeterminate through "
                f"{CAPTURE_TIMEOUT_LADDER_S[-1] * 1e3:g} ms"
            )
        incident_speed = float(sample.capture_velocity_m_per_s * factor)
        if incident_speed <= 0.0:
            raise RuntimeError(f"no positive capture boundary for atom {atom_index}")
        classification = classify_capture_trajectory(
            point, incident_speed, search_for_atom, coil_config=coil, config=config,
            model=model, beams=beams,
        )
        if classification.termination_reason not in {
            "two_core_entries", "bounded_core_residence", "escaped"
        }:
            raise RuntimeError(
                f"atom {atom_index} was indeterminate: {classification.termination_reason}"
            )
        initial_velocity = incident_speed * np.asarray(point.incident_unit_vector)
        record = simulate_rate_equation_trajectory(
            RateEquationAtomState(point.initial_position_m, tuple(initial_velocity)),
            search_for_atom.maximum_simulation_time_s,
            coil,
            beams=beams,
            model=model,
            config=config,
            trajectory_config=RateEquationTrajectoryConfig(
                time_step_s=search_for_atom.time_step_s,
                escape_radius_m=search_for_atom.escape_radius_m,
            ),
        )
        summary = trajectory_summary(record)
        label = "trapped" if classification.trapped else "escaped"
        save_trajectory(
            record, model, beams, trajectories / atom_stem,
            metadata={
                "atom_index": atom_index,
                "classification": asdict(classification),
                "capture_sample": asdict(sample),
                "search": asdict(search_for_atom),
                "coil": asdict(coil),
                "mot": asdict(config),
            },
        )
        fig = plot_trajectory_3d(
            record, beams, figures / f"{atom_stem}_trajectory_3d.png",
            title=f"Atom {atom_index:02d}: {label}; v={incident_speed:.2f} m/s; vc≈{sample.capture_velocity_m_per_s:.2f} m/s",
        )
        plt.close(fig)
        fig = plot_time_diagnostics(
            record, model, beams, figures / f"{atom_stem}_time_diagnostics.png",
        )
        plt.close(fig)
        row = {
            "atom": atom_index,
            "classification": label,
            "classification_reason": classification.termination_reason,
            "capture_max_time_ms": 1.0e3 * search_for_atom.maximum_simulation_time_s,
            "capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
            "capture_upper_m_per_s": sample.untrapped_velocity_upper_m_per_s,
            "incident_speed_m_per_s": incident_speed,
            "impact_parameter_mm": 1.0e3 * point.s_m,
            "initial_x_mm": 1.0e3 * point.initial_position_m[0],
            "initial_y_mm": 1.0e3 * point.initial_position_m[1],
            "initial_z_mm": 1.0e3 * point.initial_position_m[2],
            "initial_vx_m_per_s": float(initial_velocity[0]),
            "initial_vy_m_per_s": float(initial_velocity[1]),
            "initial_vz_m_per_s": float(initial_velocity[2]),
            "classification_time_ms": 1.0e3 * classification.elapsed_time_s,
            "trajectory_time_ms": 1.0e3 * summary["elapsed_s"],
            "final_radius_mm": 1.0e3 * summary["final_radius_m"],
            "final_speed_m_per_s": summary["final_speed_m_per_s"],
            "minimum_radius_mm": 1.0e3 * summary["minimum_radius_m"],
            "maximum_optical_force_zN": 1.0e21 * summary["maximum_optical_force_n"],
            "mean_spontaneous_rate_per_s": summary["mean_spontaneous_scattering_rate_per_s"],
        }
        _write_json(statistics / f"{atom_stem}_summary.json", row)
        rows.append(row)
        samples.append(sample)
        records.append(record)
        if classification.trapped and first_captured_record is None:
            first_captured_record = record
        print(
            f"Atom {atom_index:02d}: v={incident_speed:.3f} m/s, "
            f"vc=[{sample.capture_velocity_m_per_s:.3f}, "
            f"{sample.untrapped_velocity_upper_m_per_s:.3f}] m/s, {label}",
            flush=True,
        )

    with (statistics / "ten_atom_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _combined_plot(rows, records, figures / "ten_trajectories_3d.png")
    spectrum = capture_cross_section_spectrum(samples, SEARCH)
    save_capture_spectrum(spectrum, statistics)
    plot_capture_cross_section(spectrum, figures)
    loading = calculate_loading_rate_from_spectrum(
        np.asarray([item.velocity_m_per_s for item in spectrum]),
        np.asarray([item.capture_cross_section_m2 for item in spectrum]),
    )
    save_loading_rate_result(loading, statistics / "loading_rate_result.json")
    if first_captured_record is None:
        first_captured_record = records[0]
    movie = create_population_histogram_animation(
        first_captured_record, model,
        figures / "captured_atom_population_24_state.gif",
        max_frames=100, fps=12,
    )
    plt.close(movie._fig)
    _write_json(
        statistics / "campaign_summary.json",
        {
            "atom_count": len(rows),
            "trapped_count": sum(row["classification"] == "trapped" for row in rows),
            "escaped_count": sum(row["classification"] == "escaped" for row in rows),
            "loading_rate_atoms_per_s_illustrative": loading.loading_rate_atoms_per_s,
            "loading_warning": "ten direction discs with one impact point each are not a converged cross section or loading rate",
            "population_animation": str(figures / "captured_atom_population_24_state.gif"),
        },
    )
    print("Finished ten-atom campaign.", flush=True)


if __name__ == "__main__":
    main()
