"""Model-neutral loading-rate calculation from capture-cross-section data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np

from .configuration import RB87_MASS_KG


BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
TORR_TO_PASCAL = 133.32
DEFAULT_VAPOR_TEMPERATURE_K = 296.483
DEFAULT_RB87_PARTIAL_PRESSURE_TORR = 1.0e-10


@dataclass(frozen=True, slots=True)
class VaporLoadingParameters:
    """Rb-87 vapor assumptions used to normalize a capture spectrum.

    ``rb87_partial_pressure_torr`` is the Rb-87 partial pressure, not the
    chamber's total residual-gas pressure.  Keeping it explicit makes the
    absolute loading-rate normalization easy to change without rerunning any
    capture trajectories.
    """

    temperature_k: float = DEFAULT_VAPOR_TEMPERATURE_K
    rb87_partial_pressure_torr: float = DEFAULT_RB87_PARTIAL_PRESSURE_TORR
    rb87_mass_kg: float = RB87_MASS_KG

    def __post_init__(self) -> None:
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")
        if self.rb87_partial_pressure_torr < 0.0:
            raise ValueError("rb87_partial_pressure_torr must be nonnegative")
        if self.rb87_mass_kg <= 0.0:
            raise ValueError("rb87_mass_kg must be positive")

    @property
    def rb87_partial_pressure_pa(self) -> float:
        return self.rb87_partial_pressure_torr * TORR_TO_PASCAL

    @property
    def number_density_m3(self) -> float:
        return self.rb87_partial_pressure_pa / (
            BOLTZMANN_CONSTANT_J_PER_K * self.temperature_k
        )

    @property
    def thermal_scale_m2_per_s2(self) -> float:
        """Return ``2 k_B T / m``, the denominator scale in the exponent."""

        return (
            2.0
            * BOLTZMANN_CONSTANT_J_PER_K
            * self.temperature_k
            / self.rb87_mass_kg
        )

    @property
    def speed_distribution_prefactor_s3_per_m3(self) -> float:
        """Return the normalized 3-D Maxwell speed-distribution prefactor."""

        return 4.0 * pi * (
            self.rb87_mass_kg
            / (2.0 * pi * BOLTZMANN_CONSTANT_J_PER_K * self.temperature_k)
        ) ** 1.5

    @property
    def loading_rate_prefactor(self) -> float:
        return self.number_density_m3 * self.speed_distribution_prefactor_s3_per_m3


DEFAULT_VAPOR_LOADING_PARAMETERS = VaporLoadingParameters()
SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3 = (
    DEFAULT_VAPOR_LOADING_PARAMETERS.speed_distribution_prefactor_s3_per_m3
)
LOADING_RATE_PREFACTOR = DEFAULT_VAPOR_LOADING_PARAMETERS.loading_rate_prefactor
# Backward-compatible alias for the original misspelled public constant.
LOADING_RATE_PREFATOR = LOADING_RATE_PREFACTOR
THERMAL_SCALE_M2_PER_S2 = DEFAULT_VAPOR_LOADING_PARAMETERS.thermal_scale_m2_per_s2


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
    vapor_temperature_k: float = DEFAULT_VAPOR_TEMPERATURE_K
    rb87_partial_pressure_torr: float = DEFAULT_RB87_PARTIAL_PRESSURE_TORR
    rb87_partial_pressure_pa: float = (
        DEFAULT_VAPOR_LOADING_PARAMETERS.rb87_partial_pressure_pa
    )
    rb87_number_density_m3: float = DEFAULT_VAPOR_LOADING_PARAMETERS.number_density_m3
    rb87_mass_kg: float = RB87_MASS_KG
    speed_distribution_prefactor_s3_per_m3: float = (
        SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3
    )
    thermal_scale_m2_per_s2: float = THERMAL_SCALE_M2_PER_S2
    loading_rate_prefactor: float = LOADING_RATE_PREFACTOR

    @property
    def integral_value_m6_per_s4(self) -> float:
        """Correctly dimensioned alias for the retained legacy field name."""

        return self.integral_value_m5_per_s4


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
    vapor_parameters: VaporLoadingParameters | None = None,
) -> np.ndarray:
    """Return the tabulated integrand for the loading-rate formula."""

    vapor = vapor_parameters or DEFAULT_VAPOR_LOADING_PARAMETERS
    return capture_cross_section_m2 * velocity_m_per_s**3 * np.exp(
        -(velocity_m_per_s**2) / vapor.thermal_scale_m2_per_s2
    )


def calculate_loading_rate_from_spectrum(
    velocity_m_per_s: np.ndarray,
    capture_cross_section_m2: np.ndarray,
    vapor_parameters: VaporLoadingParameters | None = None,
) -> LoadingRateResult:
    """Numerically integrate the loading-rate formula from tabulated data."""

    if velocity_m_per_s.ndim != 1 or capture_cross_section_m2.ndim != 1:
        raise ValueError("velocity and cross-section arrays must be one-dimensional")
    if len(velocity_m_per_s) != len(capture_cross_section_m2):
        raise ValueError("velocity and cross-section arrays must have the same length")
    if len(velocity_m_per_s) < 2:
        raise ValueError("at least two spectrum samples are required")

    vapor = vapor_parameters or DEFAULT_VAPOR_LOADING_PARAMETERS
    integrand = loading_integrand(
        velocity_m_per_s,
        capture_cross_section_m2,
        vapor,
    )
    integral_value = float(np.trapezoid(integrand, velocity_m_per_s))
    # Use the correctly spelled canonical name internally.  Keep
    # ``LOADING_RATE_PREFATOR`` only as a compatibility alias for callers that
    # imported the original typo.
    loading_rate = vapor.loading_rate_prefactor * integral_value
    return LoadingRateResult(
        loading_rate_atoms_per_s=loading_rate,
        integral_value_m5_per_s4=integral_value,
        velocity_min_m_per_s=float(np.min(velocity_m_per_s)),
        velocity_max_m_per_s=float(np.max(velocity_m_per_s)),
        sample_count=int(len(velocity_m_per_s)),
        nonzero_cross_section_count=int(np.count_nonzero(capture_cross_section_m2 > 0.0)),
        quadrature_method="trapezoid",
        vapor_temperature_k=vapor.temperature_k,
        rb87_partial_pressure_torr=vapor.rb87_partial_pressure_torr,
        rb87_partial_pressure_pa=vapor.rb87_partial_pressure_pa,
        rb87_number_density_m3=vapor.number_density_m3,
        rb87_mass_kg=vapor.rb87_mass_kg,
        speed_distribution_prefactor_s3_per_m3=(
            vapor.speed_distribution_prefactor_s3_per_m3
        ),
        thermal_scale_m2_per_s2=vapor.thermal_scale_m2_per_s2,
        loading_rate_prefactor=vapor.loading_rate_prefactor,
    )


def save_loading_rate_result(result: LoadingRateResult, output_path: Path) -> Path:
    """Save a loading-rate result to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
                "integral_value_m5_per_s4": result.integral_value_m5_per_s4,
                "integral_value_m6_per_s4": result.integral_value_m6_per_s4,
                "integral_units_note": (
                    "integral_value_m5_per_s4 is a deprecated key retained for "
                    "compatibility; the correct units are m6/s4"
                ),
                "velocity_min_m_per_s": result.velocity_min_m_per_s,
                "velocity_max_m_per_s": result.velocity_max_m_per_s,
                "sample_count": result.sample_count,
                "nonzero_cross_section_count": result.nonzero_cross_section_count,
                "quadrature_method": result.quadrature_method,
                "vapor_temperature_k": result.vapor_temperature_k,
                "rb87_partial_pressure_torr": result.rb87_partial_pressure_torr,
                "rb87_partial_pressure_pa": result.rb87_partial_pressure_pa,
                "rb87_number_density_m3": result.rb87_number_density_m3,
                "rb87_mass_kg": result.rb87_mass_kg,
                "speed_distribution_prefactor_s3_per_m3": (
                    result.speed_distribution_prefactor_s3_per_m3
                ),
                "thermal_scale_m2_per_s2": result.thermal_scale_m2_per_s2,
                "loading_rate_prefactor": result.loading_rate_prefactor,
                "pressure_interpretation": "Rb-87 partial pressure",
                "speed_distribution": (
                    f"f(v) = {result.speed_distribution_prefactor_s3_per_m3:.12g} "
                    f"* v^2 * exp(-v^2 / {result.thermal_scale_m2_per_s2:.12g})"
                ),
                "formula": (
                    f"R = {result.loading_rate_prefactor:.12g} * integral "
                    "sigma_cap(v) * v^3 * "
                    f"exp(-v^2 / {result.thermal_scale_m2_per_s2:.12g}) dv"
                ),
            },
            handle,
            indent=2,
        )
    return output_path


__all__ = [
    "LOADING_RATE_PREFATOR",
    "LOADING_RATE_PREFACTOR",
    "BOLTZMANN_CONSTANT_J_PER_K",
    "DEFAULT_RB87_PARTIAL_PRESSURE_TORR",
    "DEFAULT_VAPOR_LOADING_PARAMETERS",
    "DEFAULT_VAPOR_TEMPERATURE_K",
    "SPEED_DISTRIBUTION_PREFACTOR_S3_PER_M3",
    "THERMAL_SCALE_M2_PER_S2",
    "TORR_TO_PASCAL",
    "LoadingRateResult",
    "VaporLoadingParameters",
    "calculate_loading_rate_from_spectrum",
    "load_capture_spectrum",
    "loading_integrand",
    "save_loading_rate_result",
]
