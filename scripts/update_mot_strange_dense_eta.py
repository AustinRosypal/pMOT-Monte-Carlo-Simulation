"""Forecast the remaining excluded dense diagnostic from paired measured costs."""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from plan_mot_strange_loading_production import read_json
from run_mot_strange_loading_pilot import NAME, ROOT, now, write_json


def main():
    base = Path(os.environ["LOCALAPPDATA"]) / "pmot-campaigns" / NAME
    manifest = read_json(ROOT / "pilot_manifest.json")
    launch = read_json(ROOT / "pilot_launch.json")
    workers = launch["workers"]
    resumed = datetime.fromisoformat(launch.get("timing_reference_resume_utc", launch["uv_started_utc"]).replace("Z", "+00:00"))
    excluded_intervals = set(launch.get("timing_excluded_configurations", []))
    completed, ratios, timings, remaining = set(), [], [], []
    expected_rays = manifest["discs"] * manifest["points_per_disc"]
    for index in range(len(manifest["points"])):
        folder = f"configuration_{index:03d}"
        production = read_json(base / "sampled_production" / folder / "summary.json")
        sampled_cost = production["worker_seconds"] / production["rays"]
        files = list((base / "pilot" / folder).glob("disc_*_point_*.npz"))
        summary = ROOT / "pilot" / folder / "checkpoint_summary.json"
        if summary.is_file():
            dense = read_json(summary)
            if len(files) != expected_rays or len(dense["rows"]) != expected_rays:
                raise ValueError(f"Incomplete published dense setting {index}")
            cost = sum(row[2] for row in dense["rows"])
            ratios.append(cost / expected_rays / sampled_cost)
            completed.add(index)
            end = datetime.fromtimestamp(max(p.stat().st_mtime for p in files), timezone.utc)
            timings.append((end, cost, index))
        else:
            if len(files) > expected_rays:
                raise ValueError(f"Excess dense checkpoints for setting {index}")
            remaining.append(dict(configuration=index, pending_rays=expected_rays - len(files),
                                  sampled_equivalent_worker_seconds=(expected_rays - len(files)) * sampled_cost))
    utilization, intervals = [], []
    for previous, current in zip(sorted(timings), sorted(timings)[1:]):
        # Exclude the resumed partial setting and intervals from older worker pools.
        if previous[0] < resumed or current[2] in excluded_intervals:
            continue
        seconds = (current[0] - previous[0]).total_seconds()
        value = current[1] / (workers * seconds)
        if not 0 < value <= 1.01:
            raise ValueError("Incompatible dense timing interval")
        utilization.append(min(value, 1.))
        intervals.append(dict(configuration=current[2], wall_seconds=seconds,
                              busy_seconds=current[1] / workers,
                              overhead_seconds=max(0., seconds - current[1] / workers)))
    if len(ratios) < 3 or len(utilization) < 3:
        raise ValueError("Insufficient paired timings for a measured dense forecast")
    effective_utilization = median(utilization[-6:])
    overhead = median(row["overhead_seconds"] for row in intervals)
    equivalent = sum(row["sampled_equivalent_worker_seconds"] for row in remaining)
    began = datetime.now(timezone.utc)
    estimates = {label: began + timedelta(seconds=equivalent * factor / workers + len(remaining) * extra)
                 for label, factor, extra in (
                     ("central", median(ratios), overhead),
                     ("observed_fast", min(ratios), min(row["overhead_seconds"] for row in intervals)),
                     ("observed_slow", max(ratios), max(row["overhead_seconds"] for row in intervals)))}
    affine_errors, utilization_errors = [], []
    for index, row in enumerate(intervals):
        training = intervals[:index] + intervals[index + 1:]
        affine = row["busy_seconds"] + median(r["overhead_seconds"] for r in training)
        old = row["busy_seconds"] / median(r["busy_seconds"] / r["wall_seconds"] for r in training[-6:])
        affine_errors.append(abs(affine - row["wall_seconds"]))
        utilization_errors.append(abs(old - row["wall_seconds"]))
    result = dict(status="measured_diagnostic_projection", updated_utc=now(), workers=workers,
                  completed_settings=len(completed), expected_settings=len(manifest["points"]),
                  remaining_rays=sum(row["pending_rays"] for row in remaining), paired_timings=len(ratios),
                  recent_utilization=effective_utilization, utilization_intervals=len(utilization),
                  excluded_interrupted_wall_intervals=sorted(excluded_intervals),
                  per_remaining_setting_overhead_seconds=overhead, measured_intervals=intervals,
                  leave_one_setting_out_mae_seconds=dict(
                      busy_time_plus_overhead=sum(affine_errors) / len(affine_errors),
                      recent_median_utilization=sum(utilization_errors) / len(utilization_errors)),
                  dense_to_sampled_worker_cost_ratios=dict(median=median(ratios), minimum=min(ratios), maximum=max(ratios)),
                  finish_utc={label: date.isoformat() for label, date in estimates.items()},
                  finish_central_time={label: date.astimezone(ZoneInfo("America/Chicago")).isoformat()
                                       for label, date in estimates.items()},
                  pending=remaining,
                  method="Scale remaining per-setting measured production costs by observed dense/sample per-ray ratios, divide by workers, then add the measured median per-setting overhead. Overhead includes startup and idle-worker/straggler time. A full overhead is conservatively included for a partial setting.",
                  sensitivity="Fast/slow scenarios use minimum/maximum observed paired cost ratios and setting overheads; they are not probability bounds. Future regimes can differ.",
                  scope="Excluded dense diagnostic checkpointing only; further numerical reviews, replay and final verification are not included. Production checkpointing is already complete.")
    write_json(ROOT / "dense_eta_measured.json", result)
    print({key: result[key] for key in ("completed_settings", "remaining_rays", "recent_utilization", "finish_central_time")})


if __name__ == "__main__":
    main()
