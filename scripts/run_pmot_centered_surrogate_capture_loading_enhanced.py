"""Run the enhanced centered-surrogate pMOT capture/loading campaign.

The design increases the number of statistically independent full-sphere
direction discs from 25 to 100 while retaining ten independent uniform-area
points per disc.  Exact trajectories from the earlier 25-by-10 campaign are
reused only where the launch state, speed, dynamics, and effective field are
identical; all added directions and velocity nodes are newly simulated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pmot.mot_multilevel.configuration import default_multilevel_mot_config
from pmot.mot_multilevel.simulation import build_multilevel_mot_beams
from pmot.pmot.surrogate_capture_campaign import (
    CampaignConfig,
    build_launch_tasks,
    run_campaign,
    run_representative_audits,
)
from pmot.pmot.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)


RUN_NAME = "centered field enhanced full-sphere capture loading 100x10 r15mm 20261002"
SOURCE_RUN_NAME = "centered field full-sphere capture loading 25x10 r15mm 20261002"

CAMPAIGN = CampaignConfig(
    seed=20261002,
    direction_discs=100,
    points_per_disc=10,
    launch_radius_m=10.0e-3,
    disc_radius_m=15.0e-3,
    # Two-metre-per-second resolution through the observed capture band.  The
    # 35 and 40 m/s nodes verify closure of the high-speed tail.
    speeds_m_per_s=tuple(float(value) for value in range(1, 36, 2)) + (40.0,),
    duration_s=60.0e-3,
    time_step_s=5.0e-6,
    workers=16,
    # Preserve the user's twelve-hour ceiling.  Stop starting new trajectories
    # after 10.5 h, leaving time to drain, aggregate, plot, and audit.
    submission_budget_s=10.5 * 3600.0,
    absolute_wall_budget_s=12.0 * 3600.0,
)


def _write_preflight_metadata(output_dir: Path) -> None:
    mot = default_multilevel_mot_config()
    beams = build_multilevel_mot_beams(config=mot)
    cooling = [beam for beam in beams if beam.family == "cooling"]
    repump = [beam for beam in beams if beam.family == "repump"]
    if len(cooling) != 6 or len(repump) != 6:
        raise RuntimeError("expected six cooling and six repump traveling components")
    if not all(np.isclose(beam.power_w, 27.0e-3) for beam in cooling):
        raise RuntimeError("the requested 27 mW cooling power is not active")
    if not all(np.isclose(2.0 * beam.beam_radius_m, 12.7e-3) for beam in cooling):
        raise RuntimeError("the requested 12.7 mm cooling-beam diameter is not active")
    if not np.isclose(CAMPAIGN.disc_radius_m, 15.0e-3):
        raise RuntimeError("the requested 15 mm sampling-disc radius is not active")

    payload = {
        "schema": "pmot.centered-surrogate-capture-loading-request.v2",
        "campaign": asdict(CAMPAIGN),
        "mot_configuration": asdict(mot),
        "cooling_beams": [
            {
                "label": beam.label,
                "power_w": beam.power_w,
                "one_over_e2_diameter_m": 2.0 * beam.beam_radius_m,
                "wavelength_m": beam.wavelength_m,
                "detuning_hz": beam.detuning_hz,
                "polarization": beam.circular_polarization,
            }
            for beam in cooling
        ],
        "repump_beams": [
            {
                "label": beam.label,
                "power_w": beam.power_w,
                "one_over_e2_diameter_m": 2.0 * beam.beam_radius_m,
                "wavelength_m": beam.wavelength_m,
                "detuning_hz": beam.detuning_hz,
                "polarization": beam.circular_polarization,
            }
            for beam in repump
        ],
        "effective_field_configuration": DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.metadata(),
        "sampling_rationale": (
            "Increase independent direction clusters from 25 to 100 because "
            "direction discs, rather than points within a disc, define the "
            "Student-t uncertainty. Retain 10 uniform-area points per disc."
        ),
        "method": (
            "direct velocity mask; full-sphere direction discs; uniform-area "
            "points; direction discs are independent Student-t clusters"
        ),
        "scientific_status": (
            "preliminary stretched-transition-equivalent surrogate-field "
            "diagnostic; not a state-resolved pMOT prediction"
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "requested_configuration.json"
    encoded = json.dumps(payload, indent=2)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise RuntimeError("saved preflight configuration does not match this run")
    path.write_text(encoded, encoding="utf-8")


def _seed_exact_prior_results(output_dir: Path, source_dir: Path) -> None:
    """Copy only exact overlapping deterministic trajectories into a new run."""

    destination_csv = output_dir / "trajectory_results.csv"
    if destination_csv.exists():
        return
    source_csv = source_dir / "trajectory_results.csv"
    source_manifest_path = source_dir / "manifest.json"
    if not source_csv.exists() or not source_manifest_path.exists():
        raise RuntimeError("the exact prior campaign needed for reuse is missing")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    exact_fields = (
        "seed",
        "points_per_disc",
        "launch_radius_m",
        "disc_radius_m",
        "duration_s",
        "time_step_s",
        "core_radius_m",
        "core_residence_s",
        "off_origin_window_s",
        "off_origin_centroid_min_radius_m",
        "off_origin_max_excursion_m",
        "off_origin_max_rms_speed_m_per_s",
        "off_origin_max_net_displacement_m",
    )
    current = asdict(CAMPAIGN)
    mismatches = [name for name in exact_fields if source_manifest.get(name) != current[name]]
    if mismatches:
        raise RuntimeError(f"prior campaign differs in required fields: {mismatches}")
    if source_manifest.get("effective_field_configuration") != (
        DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.metadata()
    ):
        raise RuntimeError("prior campaign used a different effective field")
    if int(source_manifest["direction_discs"]) > CAMPAIGN.direction_discs:
        raise RuntimeError("prior direction set is not a prefix of the enhanced design")

    target_tasks = {task.key: task for task in build_launch_tasks(CAMPAIGN)}
    selected_rows: list[dict[str, str]] = []
    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise RuntimeError("prior result table has no header")
        for row in reader:
            task = target_tasks.get(row["task_key"])
            if task is None:
                continue
            expected_position = np.asarray(task.initial_position_m)
            saved_position = np.asarray(
                [float(row["initial_x_m"]), float(row["initial_y_m"]), float(row["initial_z_m"])]
            )
            expected_direction = np.asarray(task.incident_unit_vector)
            saved_direction = np.asarray(
                [float(row["incident_x"]), float(row["incident_y"]), float(row["incident_z"])]
            )
            if not (
                np.allclose(saved_position, expected_position, rtol=0.0, atol=1.0e-15)
                and np.allclose(saved_direction, expected_direction, rtol=0.0, atol=1.0e-15)
                and float(row["speed_m_per_s"]) == task.speed_m_per_s
            ):
                raise RuntimeError(f"saved launch state mismatch for {task.key}")
            selected_rows.append(row)

    history_dir = output_dir / "histories"
    history_dir.mkdir(parents=True, exist_ok=True)
    with destination_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)
    for row in selected_rows:
        shutil.copy2(
            source_dir / "histories" / row["history_file"],
            history_dir / row["history_file"],
        )

    digest = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    provenance = {
        "schema": "pmot.exact-trajectory-reuse.v1",
        "source_directory": str(source_dir),
        "source_trajectory_results_sha256": digest,
        "reused_trajectories": len(selected_rows),
        "criterion": (
            "exact matching task key, launch position, incident direction, speed, "
            "trajectory duration/timestep, classification thresholds, and field configuration"
        ),
    }
    (output_dir / "reused_trajectory_provenance.json").write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(f"Reused {len(selected_rows)} exact trajectories from the 25x10 campaign.", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign_parent = (
        root
        / "outputs"
        / "diagnostics"
        / "pmot"
        / "MOT testing with a strange defined magnetic field"
    )
    output_dir = campaign_parent / RUN_NAME
    source_dir = campaign_parent / SOURCE_RUN_NAME
    _write_preflight_metadata(output_dir)
    _seed_exact_prior_results(output_dir, source_dir)
    run_campaign(output_dir, CAMPAIGN)
    run_representative_audits(output_dir, CAMPAIGN)


if __name__ == "__main__":
    main()
