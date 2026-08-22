"""Loading-rate calculation for full multilevel-MOT capture spectra."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path

import numpy as np

from ..mot_simple.loading import (
    LOADING_RATE_PREFATOR,
    LOADING_RATE_PREFACTOR,
    SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3,
    THERMAL_SCALE_M2_PER_S2,
    LoadingRateResult,
    calculate_loading_rate_from_spectrum,
    load_capture_spectrum,
    loading_integrand,
    save_loading_rate_result,
)
from .configuration import multilevel_mot_paths


@dataclass(frozen=True, slots=True)
class LoadingRateSamplingUncertainty:
    """Monte Carlo uncertainty from the hierarchical disc/point design."""

    point_standard_error_atoms_per_s: float
    point_95_percent_half_width_atoms_per_s: float
    disc_cluster_standard_error_atoms_per_s: float
    disc_cluster_95_percent_half_width_atoms_per_s: float
    disc_count: int
    point_count: int


def calculate_sampling_uncertainty(
    capture_velocity_m_per_s: np.ndarray,
    disc_index: np.ndarray,
    spectrum_velocity_m_per_s: np.ndarray,
    disc_radius_m: float,
) -> LoadingRateSamplingUncertainty:
    """Calculate point-level and direction-clustered Monte Carlo errors.

    The disc-cluster value is the appropriate primary uncertainty here because
    all impact points on one disc share the same sampled incident direction.
    """

    thresholds = np.asarray(capture_velocity_m_per_s, dtype=float)
    disc_ids = np.asarray(disc_index, dtype=int)
    velocity = np.asarray(spectrum_velocity_m_per_s, dtype=float)
    if len(thresholds) == 0 or len(thresholds) != len(disc_ids):
        raise ValueError("capture velocities and disc indices must be nonempty and aligned")
    kernel = velocity**3 * np.exp(-velocity**2 / THERMAL_SCALE_M2_PER_S2)
    area = pi * disc_radius_m**2
    contributions = np.asarray([
        LOADING_RATE_PREFATOR * np.trapezoid(
            area * (velocity <= threshold + 1.0e-12) * kernel,
            velocity,
        )
        for threshold in thresholds
    ])
    unique_discs = np.unique(disc_ids)
    disc_means = np.asarray([np.mean(contributions[disc_ids == value]) for value in unique_discs])
    point_se = float(np.std(contributions, ddof=1) / np.sqrt(len(contributions))) if len(contributions) > 1 else 0.0
    disc_se = float(np.std(disc_means, ddof=1) / np.sqrt(len(disc_means))) if len(disc_means) > 1 else 0.0
    return LoadingRateSamplingUncertainty(
        point_standard_error_atoms_per_s=point_se,
        point_95_percent_half_width_atoms_per_s=1.96 * point_se,
        disc_cluster_standard_error_atoms_per_s=disc_se,
        disc_cluster_95_percent_half_width_atoms_per_s=1.96 * disc_se,
        disc_count=len(unique_discs),
        point_count=len(thresholds),
    )


def load_capture_thresholds(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no rows found in {csv_path}")
    return (
        np.asarray([float(row["capture_velocity_m_per_s"]) for row in rows]),
        np.asarray([int(row["disc_index"]) for row in rows]),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    base = multilevel_mot_paths()["statistics"] / "loading_rate_50_discs_25_points"
    parser = argparse.ArgumentParser(
        description="Calculate full multilevel-MOT loading rate from capture-cross-section data"
    )
    parser.add_argument("--spectrum-csv", type=Path, default=base / "capture_velocity_spectrum.csv")
    parser.add_argument("--samples-csv", type=Path, default=base / "capture_velocity_samples.csv")
    parser.add_argument("--disc-radius-mm", type=float, default=12.0)
    parser.add_argument("--output-json", type=Path, default=base / "loading_rate_result.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    velocity, cross_section = load_capture_spectrum(args.spectrum_csv)
    result = calculate_loading_rate_from_spectrum(velocity, cross_section)
    thresholds, disc_indices = load_capture_thresholds(args.samples_csv)
    uncertainty = calculate_sampling_uncertainty(
        thresholds, disc_indices, velocity, 1e-3 * args.disc_radius_mm,
    )
    payload = {
        "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
        "integral_value_m5_per_s4": result.integral_value_m5_per_s4,
        "velocity_min_m_per_s": result.velocity_min_m_per_s,
        "velocity_max_m_per_s": result.velocity_max_m_per_s,
        "sample_count": result.sample_count,
        "nonzero_cross_section_count": result.nonzero_cross_section_count,
        "quadrature_method": result.quadrature_method,
        "speed_distribution": "f(v) = 2.80e-7 * v^2 * exp(-v^2 / 5.667e4)",
        "formula": "R = 9.1196e5 * integral sigma_cap(v) * v^3 * exp(-v^2 / 5.667e4) dv",
        "disc_radius_m": 1e-3 * args.disc_radius_mm,
        "sampling_uncertainty": asdict(uncertainty),
        "primary_uncertainty": "disc_cluster_standard_error_atoms_per_s",
        "spectrum_csv": str(args.spectrum_csv),
        "samples_csv": str(args.samples_csv),
        "output_json": str(args.output_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LOADING_RATE_PREFATOR",
    "LOADING_RATE_PREFACTOR",
    "SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3",
    "THERMAL_SCALE_M2_PER_S2",
    "LoadingRateResult",
    "LoadingRateSamplingUncertainty",
    "calculate_sampling_uncertainty",
    "calculate_loading_rate_from_spectrum",
    "load_capture_spectrum",
    "loading_integrand",
    "save_loading_rate_result",
]
