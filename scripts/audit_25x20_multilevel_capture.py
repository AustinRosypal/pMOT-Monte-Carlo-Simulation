"""Representative half-step/longer-window audit of the 25x20 campaign."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import (
    build_rate_equation_model,
    classify_capture_trajectory,
    default_multilevel_mot_config,
    generate_capture_launches,
    multilevel_mot_paths,
)

from run_25x20_multilevel_capture import RUN_NAME, SEARCH, TRAPPED_REASONS


def _worker(payload):
    point, source = payload
    sample = source["sample"]
    config = default_multilevel_mot_config()
    coil = default_anti_helmholtz_config()
    audit_search = replace(
        SEARCH,
        time_step_s=0.5 * SEARCH.time_step_s,
        maximum_simulation_time_s=max(40.0e-3, 2.0 * float(source["maximum_simulation_time_s"])),
    )
    speeds = {
        "lower": float(sample["trapped_velocity_lower_m_per_s"]),
        "upper": float(sample["untrapped_velocity_upper_m_per_s"]),
        "low_speed": min(5.0, 0.5 * float(sample["trapped_velocity_lower_m_per_s"])),
    }
    results = {}
    for name, speed in speeds.items():
        outcome = classify_capture_trajectory(
            point, speed, audit_search, config=config, coil_config=coil,
        )
        results[name] = {
            "speed_m_per_s": speed,
            "trapped": outcome.trapped,
            "reason": outcome.termination_reason,
            "elapsed_s": outcome.elapsed_time_s,
        }
    endpoints_agree = (
        results["lower"]["trapped"]
        and results["lower"]["reason"] in TRAPPED_REASONS
        and not results["upper"]["trapped"]
        and results["upper"]["reason"] == "escaped"
    )
    low_speed_agrees = results["low_speed"]["trapped"]
    return {
        "disc_index": point.disc_index,
        "point_index": point.point_index,
        "impact_parameter_m": point.s_m,
        "original_capture_lower_m_per_s": speeds["lower"],
        "original_capture_upper_m_per_s": speeds["upper"],
        "audit_time_step_s": audit_search.time_step_s,
        "audit_maximum_simulation_time_s": audit_search.maximum_simulation_time_s,
        "classifications": results,
        "endpoints_agree": endpoints_agree,
        "low_speed_agrees_with_scalar_assumption": low_speed_agrees,
    }


def main() -> None:
    statistics = multilevel_mot_paths()["statistics"] / RUN_NAME
    manifest = json.loads((statistics / "capture_run_config.json").read_text(encoding="utf-8"))
    if not np.isclose(manifest["mot"]["cooling_power_w_per_beam"], 27.0e-3):
        raise RuntimeError("not the 27 mW campaign")
    discs, points = generate_capture_launches(SEARCH)
    if len(discs) != 25 or len(points) != 500:
        raise RuntimeError("campaign geometry mismatch")
    selected = []
    for disc_index in range(25):
        disc_points = [point for point in points if point.disc_index == disc_index]
        # Alternate near-center and near-rim rays to cover distinct geometry.
        point = min(disc_points, key=lambda item: item.s_m) if disc_index % 2 == 0 else max(disc_points, key=lambda item: item.s_m)
        result_path = statistics / "rays" / f"disc_{disc_index:02d}_point_{point.point_index:02d}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "resolved":
            raise RuntimeError(f"audit source ray is unresolved: {result_path}")
        selected.append((point, result))
    # Preload ARC before forking to avoid concurrent SQLite cache creation.
    build_rate_equation_model()
    output = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_worker, payload) for payload in selected]
        for future in as_completed(futures):
            result = future.result()
            output.append(result)
            print(
                f"disc={result['disc_index']:02d} point={result['point_index']:02d} "
                f"endpoints={result['endpoints_agree']} low_speed={result['low_speed_agrees_with_scalar_assumption']}",
                flush=True,
            )
    output.sort(key=lambda item: item["disc_index"])
    report = {
        "schema": "pmot.mot_multilevel.capture-25x20-representative-audit.v1",
        "selection": "one per disc: smallest impact on even discs, largest impact on odd discs",
        "all_25_endpoints_agree": all(item["endpoints_agree"] for item in output),
        "all_25_low_speed_checks_agree": all(item["low_speed_agrees_with_scalar_assumption"] for item in output),
        "limitations": "does not prove global monotonicity or convergence of all 500 boundaries",
        "audits": output,
    }
    path = statistics / "representative_half_step_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
