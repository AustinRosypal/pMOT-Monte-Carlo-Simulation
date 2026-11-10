"""Compare finite retention windows for the approved symmetric field setup."""

import json
from dataclasses import asdict
from time import perf_counter

import numpy as np

from pmot.mot_strange.loading_scan_inputs import (
    ScanPoint, current_symmetric_surrogate, field_config, prepare, sample_launch_geometry,
)
from pmot.mot_strange.loading_scan_retention import (
    CANDIDATE_HORIZON_S, OBSERVATION_WINDOW_S, agreed_category, observe,
)
from validate_mot_strange_loading_scans import write_json


def record(observation):
    code, duration, position, velocity, first, anchor, min_radius, max_radius, excursion = observation
    return {
        "category": code, "duration_s": duration,
        "endpoint_m": position.tolist(), "endpoint_m_per_s": velocity.tolist(),
        "first_core_event_s": first, "window_anchor_m": anchor.tolist(),
        "window_minimum_radius_m": min_radius if np.isfinite(min_radius) else None,
        "window_maximum_radius_m": max_radius, "window_maximum_excursion_m": excursion,
    }


def main():
    geometry = current_symmetric_surrogate()
    positions, directions, _ = sample_launch_geometry(3, 2, 2026110502, geometry)
    cases = (
        ScanPoint(),
        ScanPoint("axial", 1),
        ScanPoint("axial", 40),
        ScanPoint("isotropic", 40),
        ScanPoint("axial", 10, -40),
        ScanPoint("axial", 10, -15, 5),
    )
    launches = [(positions[0, 0], directions[0]), (positions[2, 1], directions[2]),
                (np.array([0., 0., .01]), np.array([0., 0., -1.]))]
    speeds = (.25, 3., 8., 15., 25., 35., 55., 80.)
    horizons = (CANDIDATE_HORIZON_S, .8)
    checks = []
    for point in cases:
        data = prepare(point, field_config(point, geometry, isotropic_axial_helicity=-1))
        observe(launches[0][0], 80 * launches[0][1], data, .4, .1, 1.)
        start = perf_counter()
        outcomes = []
        for r, direction in launches:
            for speed in speeds:
                pairs = []
                for horizon in horizons:
                    a, b = (
                        observe(r, speed * direction, data, horizon, OBSERVATION_WINDOW_S, refinement)
                        for refinement in (1., 2.)
                    )
                    category = agreed_category(a, b)
                    pairs.append({"horizon_s": horizon, "category": category,
                                  "adaptive": [record(a), record(b)]})
                outcomes.append({"position_m": r.tolist(), "direction": direction.tolist(),
                                 "speed_m_per_s": speed, "windows": pairs})
        checks.append({"point": asdict(point), "elapsed_s": perf_counter() - start, "outcomes": outcomes})
        write_json("retention_horizon_probe.json", checks)
        unresolved = sum(w["category"] is None for x in outcomes for w in x["windows"])
        changed = sum(
            a["category"] != b["category"] for a, b in (x["windows"] for x in outcomes)
            if a["category"] is not None and b["category"] is not None
        )
        print(json.dumps(asdict(point)), len(outcomes), "nodes", unresolved, "unresolved",
              changed, "horizon changes", round(checks[-1]["elapsed_s"], 2), "seconds", flush=True)


if __name__ == "__main__":
    main()
