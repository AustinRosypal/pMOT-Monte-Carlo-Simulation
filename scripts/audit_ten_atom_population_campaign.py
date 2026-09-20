"""Check the ten-atom probe's classifications at half the trajectory step."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    CaptureSearchConfig,
    build_multilevel_mot_beams,
    build_rate_equation_model,
    classify_capture_trajectory,
    default_multilevel_mot_config,
    generate_capture_launches,
    multilevel_mot_paths,
)

from run_ten_atom_population_campaign import RUN_COOLING_POWER_W_PER_BEAM, RUN_NAME, SEARCH


def main() -> None:
    root = multilevel_mot_paths()["statistics"] / RUN_NAME
    rows = list(csv.DictReader((root / "ten_atom_summary.csv").open(encoding="utf-8")))
    if len(rows) != 10:
        raise RuntimeError("ten-atom probe is incomplete")
    points = generate_capture_launches(SEARCH)[1]
    config = replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=RUN_COOLING_POWER_W_PER_BEAM,
    )
    coil = default_anti_helmholtz_config()
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    results = []
    all_agree = True
    for row, point in zip(rows, points):
        atom = int(row["atom"])
        if atom != point.disc_index + 1:
            raise RuntimeError("row/point mismatch")
        original_max_time = 1.0e-3 * float(row["capture_max_time_ms"])
        audit = replace(
            SEARCH,
            time_step_s=0.5 * SEARCH.time_step_s,
            maximum_simulation_time_s=max(30.0e-3, 2.0 * original_max_time),
        )
        trial_speeds = {
            "selected": float(row["incident_speed_m_per_s"]),
            "lower": float(row["capture_velocity_m_per_s"]),
            "upper": float(row["capture_upper_m_per_s"]),
        }
        classifications = {}
        for name, speed in trial_speeds.items():
            result = classify_capture_trajectory(
                point, speed, audit, coil_config=coil, config=config,
                model=model, beams=beams,
            )
            classifications[name] = {
                "speed_m_per_s": speed,
                "trapped": result.trapped,
                "reason": result.termination_reason,
                "elapsed_s": result.elapsed_time_s,
                "final_radius_m": result.final_radius_m,
            }
        agrees = (
            classifications["selected"]["trapped"] == (row["classification"] == "trapped")
            and classifications["lower"]["trapped"]
            and not classifications["upper"]["trapped"]
            and classifications["upper"]["reason"] == "escaped"
        )
        all_agree &= agrees
        results.append(
            {
                "atom": atom,
                "original_classification": row["classification"],
                "original_capture_window_s": original_max_time,
                "audit_config": asdict(audit),
                "classifications": classifications,
                "all_three_agree": agrees,
            }
        )
        print(f"Atom {atom:02d}: {'PASS' if agrees else 'FAIL'} {classifications}", flush=True)
    report = {
        "schema": "pmot.mot_multilevel.ten-atom-half-step-audit.v1",
        "all_agree": all_agree,
        "note": "This checks selected speeds and saved scalar brackets; it does not prove local monotonicity or convergence of the full spectrum.",
        "atoms": results,
    }
    path = root / "half_step_capture_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}; all_agree={all_agree}", flush=True)


if __name__ == "__main__":
    main()
