"""Dense one-dimensional refinement of the equal-power pMOT optimum."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from optimize_pmot_2level_trapping_powers import AXES
from optimize_pmot_2level_trapping_powers import DENSE_COORDINATES_M
from optimize_pmot_2level_trapping_powers import PROJECT_ROOT
from optimize_pmot_2level_trapping_powers import _metrics
from optimize_pmot_2level_trapping_powers import _observable


OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "pmot_2level_trapping_power_optimization_20261023"
    / "equal_power_dense_refinement.json"
)


def _equal_power_score(power_mw: float) -> float:
    powers = np.full(3, power_mw)
    force = np.empty(len(DENSE_COORDINATES_M))
    for index, coordinate in enumerate(DENSE_COORDINATES_M):
        position = np.asarray((coordinate, 0.0, 0.0))
        force[index] = _observable(position, powers).force_n[0]
    gradient = np.gradient(force, DENSE_COORDINATES_M, edge_order=2)
    weakest_local_restoring = float(1.0e19 * np.min(-gradient))
    average_restoring = float(
        1.0e19 * (force[0] - force[-1])
        / (DENSE_COORDINATES_M[-1] - DENSE_COORDINATES_M[0])
    )
    return weakest_local_restoring + 0.02 * average_restoring


def main() -> None:
    result = minimize_scalar(
        lambda power: -_equal_power_score(power),
        bounds=(28.0, 34.0),
        method="bounded",
        options={"xatol": 1.0e-7, "maxiter": 100},
    )
    if not result.success:
        raise RuntimeError(result.message)
    power = float(result.x)
    _, metrics = _metrics(np.full(3, power), DENSE_COORDINATES_M)
    payload = {
        "equal_power_mw_per_path": power,
        "dense_objective_score": -float(result.fun),
        "origin_effective_detuning_mhz": metrics["origin"][
            "effective_detuning_mhz"
        ],
        "origin_common_ac_stark_shift_mhz": metrics["origin"][
            "common_ac_stark_shift_mhz"
        ],
        "axis_metrics": {axis: metrics[axis] for axis in AXES},
        "coordinate_grid": {
            "minimum_mm": -1.0,
            "maximum_mm": 1.0,
            "points": int(len(DENSE_COORDINATES_M)),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
