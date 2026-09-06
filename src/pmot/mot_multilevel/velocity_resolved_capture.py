"""Velocity-resolved capture aggregation for exceptional nonmonotone rays.

Production capture searches normally summarize one launch ray by a scalar
capture threshold.  A directly audited trapped band cannot be represented by
that scalar without inventing capture at lower speeds.  This module retains
the ordinary threshold estimator for every unexceptional ray and substitutes
an explicit boolean capture mask only for audited exceptional rays.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Mapping, Sequence

import numpy as np

from ..capture_statistics import CaptureVelocitySample
from ..loading import (
    LOADING_RATE_PREFACTOR,
    THERMAL_SCALE_M2_PER_S2,
    calculate_loading_rate_from_spectrum,
)
from .power_loading_study import (
    TRAPPED_TERMINATION_REASONS,
    _student_t_critical_95,
    _velocity_grid,
    calculate_clustered_cross_section,
    calculate_disc_clustered_loading,
)
from .rate_capture import RateCaptureSearchConfig


@dataclass(frozen=True, slots=True)
class VelocityResolvedCaptureOverride:
    """Direct capture classifications for one launch ray on a velocity grid."""

    disc_index: int
    point_index: int
    velocity_m_per_s: tuple[float, ...]
    captured: tuple[bool, ...]

    @property
    def key(self) -> tuple[int, int]:
        return self.disc_index, self.point_index


def validate_velocity_resolved_overrides(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    overrides: Sequence[VelocityResolvedCaptureOverride],
) -> dict[tuple[int, int], VelocityResolvedCaptureOverride]:
    """Validate overrides and return them keyed by disc and point index."""

    sample_keys = {(sample.disc_index, sample.point_index) for sample in samples}
    expected_velocity = np.arange(
        0.0,
        search.analysis_velocity_max_m_per_s
        + 0.5 * search.analysis_velocity_step_m_per_s,
        search.analysis_velocity_step_m_per_s,
        dtype=float,
    )
    keyed: dict[tuple[int, int], VelocityResolvedCaptureOverride] = {}
    for override in overrides:
        if override.key in keyed:
            raise ValueError(f"duplicate velocity-resolved override {override.key}")
        if override.key not in sample_keys:
            raise ValueError(f"override has no matching capture sample {override.key}")
        velocity = np.asarray(override.velocity_m_per_s, dtype=float)
        if (
            len(velocity) != len(expected_velocity)
            or len(override.captured) != len(expected_velocity)
            or not np.all(np.isfinite(velocity))
            or not np.allclose(
                velocity,
                expected_velocity,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError(
                f"override {override.key} does not use the exact 0-to-analysis-max grid"
            )
        if override.captured[0] or not any(override.captured):
            raise ValueError(
                f"override {override.key} is not a finite-speed capture band"
            )
        if override.captured[-1]:
            raise ValueError(
                f"override {override.key} has no escaped high-speed endpoint"
            )
        keyed[override.key] = override
    return keyed


def _capture_predicate(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    overrides: Sequence[VelocityResolvedCaptureOverride],
):
    keyed = validate_velocity_resolved_overrides(samples, search, overrides)
    indexed_masks = {
        key: {
            float(velocity): bool(captured)
            for velocity, captured in zip(
                override.velocity_m_per_s, override.captured, strict=True
            )
        }
        for key, override in keyed.items()
    }

    def captured_at(sample: CaptureVelocitySample, speed: float) -> bool:
        mask = indexed_masks.get((sample.disc_index, sample.point_index))
        if mask is not None:
            direct = next(
                (
                    captured
                    for velocity, captured in mask.items()
                    if np.isclose(speed, velocity, rtol=0.0, atol=1.0e-12)
                ),
                None,
            )
            if direct is None:
                raise ValueError(
                    "requested velocity has no direct evidence for overridden "
                    f"sample {(sample.disc_index, sample.point_index)}: {speed:g} m/s"
                )
            return direct
        return bool(
            sample.lower_classification in TRAPPED_TERMINATION_REASONS
            and sample.capture_velocity_m_per_s >= speed - 1.0e-12
        )

    return captured_at


def calculate_clustered_cross_section_with_overrides(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    overrides: Sequence[VelocityResolvedCaptureOverride],
    velocity_grid_m_per_s: np.ndarray | None = None,
) -> list[dict[str, int | float]]:
    """Calculate direction-clustered cross sections with direct ray masks."""

    if not overrides:
        return calculate_clustered_cross_section(
            samples, search, velocity_grid_m_per_s
        )
    if not samples:
        raise ValueError("at least one capture sample is required")
    velocity = (
        _velocity_grid(samples, search)
        if velocity_grid_m_per_s is None
        else np.asarray(velocity_grid_m_per_s, dtype=float)
    )
    if velocity.ndim != 1 or len(velocity) < 2 or np.any(np.diff(velocity) <= 0.0):
        raise ValueError("velocity grid must be strictly increasing and one-dimensional")
    if velocity[0] < -1.0e-12 or velocity[-1] > search.analysis_velocity_max_m_per_s + 1.0e-12:
        raise ValueError("velocity-resolved overrides do not cover the requested spectrum grid")
    captured_at = _capture_predicate(samples, search, overrides)
    grouped: dict[int, list[CaptureVelocitySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.disc_index, []).append(sample)
    disc_count = len(grouped)
    area = pi * search.disc_radius_m**2
    t_critical = _student_t_critical_95(disc_count)
    rows: list[dict[str, int | float]] = []
    for speed in velocity:
        speed_value = float(speed)
        disc_sigma = area * np.asarray(
            [
                np.mean(
                    [captured_at(sample, speed_value) for sample in grouped[index]]
                )
                for index in sorted(grouped)
            ]
        )
        mean = float(np.mean(disc_sigma))
        sample_std = float(np.std(disc_sigma, ddof=1)) if disc_count > 1 else 0.0
        sem = sample_std / np.sqrt(disc_count)
        half_width = t_critical * sem
        rows.append(
            {
                "velocity_m_per_s": speed_value,
                "captured_count": int(
                    sum(captured_at(sample, speed_value) for sample in samples)
                ),
                "launched_count": len(samples),
                "capture_fraction": mean / area,
                "capture_cross_section_m2": mean,
                "capture_cross_section_sample_std_m2": sample_std,
                "capture_cross_section_disc_cluster_sem_m2": sem,
                "capture_cross_section_t95_lower_m2": max(0.0, mean - half_width),
                "capture_cross_section_t95_upper_m2": min(area, mean + half_width),
                "disc_count": disc_count,
                "student_t_critical_95": t_critical,
            }
        )
    return rows


def calculate_disc_clustered_loading_with_overrides(
    samples: Sequence[CaptureVelocitySample],
    search: RateCaptureSearchConfig,
    spectrum_rows: Sequence[Mapping[str, int | float]],
    overrides: Sequence[VelocityResolvedCaptureOverride],
) -> tuple[list[dict[str, int | float]], dict[str, int | float | str]]:
    """Integrate each direction disc using any direct velocity masks."""

    if not overrides:
        return calculate_disc_clustered_loading(samples, search, spectrum_rows)
    velocity = np.asarray(
        [row["velocity_m_per_s"] for row in spectrum_rows], dtype=float
    )
    if len(velocity) < 2:
        raise ValueError("capture spectrum must have at least two velocity points")
    captured_at = _capture_predicate(samples, search, overrides)
    grouped: dict[int, list[CaptureVelocitySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.disc_index, []).append(sample)
    area = pi * search.disc_radius_m**2
    by_disc: list[dict[str, int | float]] = []
    for disc_index in sorted(grouped):
        disc_samples = grouped[disc_index]
        sigma = area * np.asarray(
            [
                np.mean(
                    [captured_at(sample, float(speed)) for sample in disc_samples]
                )
                for speed in velocity
            ],
            dtype=float,
        )
        result = calculate_loading_rate_from_spectrum(velocity, sigma)
        by_disc.append(
            {
                "disc_index": disc_index,
                "point_count": len(disc_samples),
                "loading_integral_m6_per_s4": result.integral_value_m5_per_s4,
                "loading_rate_atoms_per_s": result.loading_rate_atoms_per_s,
            }
        )
    rates = np.asarray(
        [row["loading_rate_atoms_per_s"] for row in by_disc], dtype=float
    )
    integrals = np.asarray(
        [row["loading_integral_m6_per_s4"] for row in by_disc], dtype=float
    )
    mean_sigma = np.asarray(
        [row["capture_cross_section_m2"] for row in spectrum_rows], dtype=float
    )
    mean_result = calculate_loading_rate_from_spectrum(velocity, mean_sigma)
    disc_count = len(by_disc)
    sample_std = float(np.std(rates, ddof=1)) if disc_count > 1 else 0.0
    sem = sample_std / np.sqrt(disc_count)
    t_critical = _student_t_critical_95(disc_count)
    mean = float(np.mean(rates))
    half_width = t_critical * sem
    return by_disc, {
        "loading_rate_mean_atoms_per_s": mean,
        "loading_rate_from_mean_spectrum_atoms_per_s": mean_result.loading_rate_atoms_per_s,
        "loading_rate_sample_std_atoms_per_s": sample_std,
        "loading_rate_disc_cluster_sem_atoms_per_s": sem,
        "loading_rate_t95_lower_atoms_per_s": max(0.0, mean - half_width),
        "loading_rate_t95_upper_atoms_per_s": mean + half_width,
        "student_t_critical_95": t_critical,
        "confidence_level": 0.95,
        "disc_count": disc_count,
        "point_count": len(samples),
        "loading_integral_mean_m6_per_s4": float(np.mean(integrals)),
        "loading_integral_from_mean_spectrum_m6_per_s4": mean_result.integral_value_m5_per_s4,
        "velocity_min_m_per_s": float(np.min(velocity)),
        "velocity_max_m_per_s": float(np.max(velocity)),
        "velocity_grid_sample_count": len(velocity),
        "quadrature_method": "trapezoid",
        "formula": (
            "R = 9.1196e5 * integral sigma_capture(v) * v^3 * "
            "exp[-v^2/(5.667e4)] dv"
        ),
        "raw_integral_units": "m^6/s^4",
        "loading_rate_prefactor": LOADING_RATE_PREFACTOR,
        "thermal_scale_m2_per_s2": THERMAL_SCALE_M2_PER_S2,
        "primary_uncertainty": (
            "disc-clustered standard error across incident directions"
        ),
    }


__all__ = [
    "VelocityResolvedCaptureOverride",
    "calculate_clustered_cross_section_with_overrides",
    "calculate_disc_clustered_loading_with_overrides",
    "validate_velocity_resolved_overrides",
]
