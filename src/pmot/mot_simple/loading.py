"""Simplified-MOT CLI for the shared loading-rate calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..loading import (
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
from .configuration import simple_mot_paths


def build_argument_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for loading-rate evaluation."""

    default_spectrum = simple_mot_paths()["outputs_statistics_simple_mot"] / "capture_velocity_spectrum.csv"
    default_output = simple_mot_paths()["outputs_statistics_simple_mot"] / "loading_rate_result.json"
    parser = argparse.ArgumentParser(description="Calculate simplified-MOT loading rate from capture-cross-section data")
    parser.add_argument("--spectrum-csv", type=Path, default=default_spectrum)
    parser.add_argument("--output-json", type=Path, default=default_output)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    velocities, cross_sections = load_capture_spectrum(args.spectrum_csv)
    result = calculate_loading_rate_from_spectrum(velocities, cross_sections)
    save_loading_rate_result(result, args.output_json)
    print(json.dumps(
        {
            "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
            "integral_value_m5_per_s4": result.integral_value_m5_per_s4,
            "velocity_min_m_per_s": result.velocity_min_m_per_s,
            "velocity_max_m_per_s": result.velocity_max_m_per_s,
            "sample_count": result.sample_count,
            "nonzero_cross_section_count": result.nonzero_cross_section_count,
            "quadrature_method": result.quadrature_method,
        },
        indent=2,
    ))
    return 0


__all__ = [
    "LOADING_RATE_PREFATOR",
    "LOADING_RATE_PREFACTOR",
    "SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3",
    "THERMAL_SCALE_M2_PER_S2",
    "LoadingRateResult",
    "build_argument_parser",
    "calculate_loading_rate_from_spectrum",
    "load_capture_spectrum",
    "loading_integrand",
    "main",
    "save_loading_rate_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
