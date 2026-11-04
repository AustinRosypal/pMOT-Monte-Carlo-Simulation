"""Simplified-MOT CLI for the shared loading-rate calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..loading import (
    DEFAULT_RB87_PARTIAL_PRESSURE_TORR,
    DEFAULT_VAPOR_TEMPERATURE_K,
    LOADING_RATE_PREFATOR,
    LOADING_RATE_PREFACTOR,
    SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3,
    THERMAL_SCALE_M2_PER_S2,
    LoadingRateResult,
    VaporLoadingParameters,
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
    parser.add_argument(
        "--temperature-k",
        type=float,
        default=DEFAULT_VAPOR_TEMPERATURE_K,
        help="Rb-87 vapor temperature in kelvin (default: %(default)s)",
    )
    parser.add_argument(
        "--rb87-pressure-torr",
        type=float,
        default=DEFAULT_RB87_PARTIAL_PRESSURE_TORR,
        help=(
            "Rb-87 partial pressure in Torr, not total chamber pressure "
            "(default: %(default)s)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    velocities, cross_sections = load_capture_spectrum(args.spectrum_csv)
    vapor = VaporLoadingParameters(
        temperature_k=args.temperature_k,
        rb87_partial_pressure_torr=args.rb87_pressure_torr,
    )
    result = calculate_loading_rate_from_spectrum(
        velocities,
        cross_sections,
        vapor,
    )
    save_loading_rate_result(result, args.output_json)
    print(json.dumps(
        {
            "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
            "integral_value_m5_per_s4": result.integral_value_m5_per_s4,
            "integral_value_m6_per_s4": result.integral_value_m6_per_s4,
            "vapor_temperature_k": result.vapor_temperature_k,
            "rb87_partial_pressure_torr": result.rb87_partial_pressure_torr,
            "rb87_number_density_m3": result.rb87_number_density_m3,
            "loading_rate_prefactor": result.loading_rate_prefactor,
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
    "DEFAULT_RB87_PARTIAL_PRESSURE_TORR",
    "DEFAULT_VAPOR_TEMPERATURE_K",
    "SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3",
    "THERMAL_SCALE_M2_PER_S2",
    "LoadingRateResult",
    "VaporLoadingParameters",
    "build_argument_parser",
    "calculate_loading_rate_from_spectrum",
    "load_capture_spectrum",
    "loading_integrand",
    "main",
    "save_loading_rate_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
