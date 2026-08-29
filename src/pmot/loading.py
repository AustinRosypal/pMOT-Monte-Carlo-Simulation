"""Model-neutral loading-rate calculation from capture-cross-section data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3 = 2.80e-7
LOADING_RATE_PREFACTOR = 9.1196e5
# Backward-compatible alias for the original misspelled public constant.
LOADING_RATE_PREFATOR = LOADING_RATE_PREFACTOR
THERMAL_SCALE_M2_PER_S2 = 5.667e4


@dataclass(frozen=True, slots=True)
class LoadingRateResult:
    """Numerical loading-rate integral result."""

    loading_rate_atoms_per_s: float
    integral_value_m5_per_s4: float
    velocity_min_m_per_s: float
    velocity_max_m_per_s: float
    sample_count: int
    nonzero_cross_section_count: int
    quadrature_method: str


def load_capture_spectrum(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a tabulated capture-cross-section spectrum."""

    velocities: list[float] = []
    cross_sections: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            velocities.append(float(row["velocity_m_per_s"]))
            cross_sections.append(float(row["capture_cross_section_m2"]))
    if not velocities:
        raise ValueError(f"no rows found in {csv_path}")
    velocity_array = np.asarray(velocities, dtype=float)
    cross_section_array = np.asarray(cross_sections, dtype=float)
    order = np.argsort(velocity_array)
    return velocity_array[order], cross_section_array[order]


def loading_integrand(
    velocity_m_per_s: np.ndarray,
    capture_cross_section_m2: np.ndarray,
) -> np.ndarray:
    """Return the tabulated integrand for the loading-rate formula."""

    return capture_cross_section_m2 * velocity_m_per_s**3 * np.exp(
        -(velocity_m_per_s**2) / THERMAL_SCALE_M2_PER_S2
    )


def calculate_loading_rate_from_spectrum(
    velocity_m_per_s: np.ndarray,
    capture_cross_section_m2: np.ndarray,
) -> LoadingRateResult:
    """Numerically integrate the loading-rate formula from tabulated data."""

    if velocity_m_per_s.ndim != 1 or capture_cross_section_m2.ndim != 1:
        raise ValueError("velocity and cross-section arrays must be one-dimensional")
    if len(velocity_m_per_s) != len(capture_cross_section_m2):
        raise ValueError("velocity and cross-section arrays must have the same length")
    if len(velocity_m_per_s) < 2:
        raise ValueError("at least two spectrum samples are required")

    integrand = loading_integrand(velocity_m_per_s, capture_cross_section_m2)
    integral_value = float(np.trapezoid(integrand, velocity_m_per_s))
    loading_rate = LOADING_RATE_PREFATOR * integral_value
    return LoadingRateResult(
        loading_rate_atoms_per_s=loading_rate,
        integral_value_m5_per_s4=integral_value,
        velocity_min_m_per_s=float(np.min(velocity_m_per_s)),
        velocity_max_m_per_s=float(np.max(velocity_m_per_s)),
        sample_count=int(len(velocity_m_per_s)),
        nonzero_cross_section_count=int(np.count_nonzero(capture_cross_section_m2 > 0.0)),
        quadrature_method="trapezoid",
    )


def save_loading_rate_result(result: LoadingRateResult, output_path: Path) -> Path:
    """Save a loading-rate result to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
                "integral_value_m5_per_s4": result.integral_value_m5_per_s4,
                "velocity_min_m_per_s": result.velocity_min_m_per_s,
                "velocity_max_m_per_s": result.velocity_max_m_per_s,
                "sample_count": result.sample_count,
                "nonzero_cross_section_count": result.nonzero_cross_section_count,
                "quadrature_method": result.quadrature_method,
                "speed_distribution": "f(v) = 2.80e-7 * v^2 * exp(-v^2 / 5.667e4)",
                "formula": (
                    "R = 9.1196e5 * integral sigma_cap(v) * v^3 * "
                    "exp(-v^2 / 5.667e4) dv"
                ),
            },
            handle,
            indent=2,
        )
    return output_path


__all__ = [
    "LOADING_RATE_PREFATOR",
    "LOADING_RATE_PREFACTOR",
    "SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3",
    "THERMAL_SCALE_M2_PER_S2",
    "LoadingRateResult",
    "calculate_loading_rate_from_spectrum",
    "load_capture_spectrum",
    "loading_integrand",
    "save_loading_rate_result",
]
