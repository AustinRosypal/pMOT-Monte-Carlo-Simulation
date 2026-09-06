from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pmot.capture_statistics import CaptureVelocitySample
from pmot.mot_multilevel.power_loading_study import (
    calculate_clustered_cross_section,
    calculate_disc_clustered_loading,
)
from pmot.mot_multilevel.rate_capture import RateCaptureSearchConfig
from pmot.mot_multilevel.velocity_resolved_capture import (
    VelocityResolvedCaptureOverride,
    calculate_clustered_cross_section_with_overrides,
    calculate_disc_clustered_loading_with_overrides,
    validate_velocity_resolved_overrides,
)


def _sample(disc: int, point: int, capture_velocity: float) -> CaptureVelocitySample:
    return CaptureVelocitySample(
        disc_index=disc,
        point_index=point,
        theta_rad=1.0,
        phi_rad=2.0,
        theta_prime_rad=0.5,
        s_m=1.0e-3,
        radial_distance_m=15.0e-3,
        initial_position_m=(15.0e-3, 0.0, 0.0),
        incident_unit_vector=(-1.0, 0.0, 0.0),
        capture_velocity_m_per_s=capture_velocity,
        velocity_resolution_m_per_s=0.25,
        trapped_velocity_lower_m_per_s=capture_velocity,
        untrapped_velocity_upper_m_per_s=capture_velocity + 0.25,
        lower_classification=(
            "bounded_core_residence" if capture_velocity > 0.0 else "escaped"
        ),
        upper_classification="escaped",
        lower_entered_trap_core=capture_velocity > 0.0,
        upper_entered_trap_core=False,
        lower_core_entry_count=1 if capture_velocity > 0.0 else 0,
        upper_core_entry_count=0,
    )


def _search() -> RateCaptureSearchConfig:
    return replace(
        RateCaptureSearchConfig(),
        disc_count=2,
        points_per_disc=2,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=2.0,
    )


def test_no_override_path_is_bit_for_bit_legacy_equivalent() -> None:
    samples = [_sample(0, 0, 1.0), _sample(0, 1, 0.5), _sample(1, 0, 1.5), _sample(1, 1, 0.0)]
    search = _search()

    legacy_spectrum = calculate_clustered_cross_section(samples, search)
    revised_spectrum = calculate_clustered_cross_section_with_overrides(
        samples, search, []
    )
    legacy_by_disc, legacy_loading = calculate_disc_clustered_loading(
        samples, search, legacy_spectrum
    )
    revised_by_disc, revised_loading = (
        calculate_disc_clustered_loading_with_overrides(
            samples, search, revised_spectrum, []
        )
    )

    assert revised_spectrum == legacy_spectrum
    assert revised_by_disc == legacy_by_disc
    assert revised_loading == legacy_loading


def test_direct_mask_preserves_a_nonmonotone_capture_band() -> None:
    samples = [_sample(0, 0, 0.0), _sample(0, 1, 1.0), _sample(1, 0, 0.5), _sample(1, 1, 0.0)]
    search = _search()
    velocity = tuple(float(value) for value in np.arange(0.0, 2.01, 0.25))
    override = VelocityResolvedCaptureOverride(
        0,
        0,
        velocity,
        tuple(value in {0.5, 0.75, 1.0} for value in velocity),
    )

    spectrum = calculate_clustered_cross_section_with_overrides(
        samples, search, [override]
    )
    by_disc, loading = calculate_disc_clustered_loading_with_overrides(
        samples, search, spectrum, [override]
    )
    captured = {
        row["velocity_m_per_s"]: row["captured_count"] for row in spectrum
    }

    assert captured[0.0] == 2
    assert captured[0.25] == 2
    assert captured[0.5] == 3
    assert captured[0.75] == 2
    assert captured[1.0] == 2
    assert captured[1.25] == 0
    assert len(by_disc) == 2
    assert loading["loading_rate_mean_atoms_per_s"] == pytest.approx(
        loading["loading_rate_from_mean_spectrum_atoms_per_s"], abs=1.0e-10
    )


def test_override_rejects_incomplete_or_unaudited_velocity_grids() -> None:
    samples = [_sample(0, 0, 0.0)]
    search = replace(_search(), disc_count=1, points_per_disc=1)
    incomplete = VelocityResolvedCaptureOverride(
        0,
        0,
        (0.0, 0.25, 0.5),
        (False, True, False),
    )

    with pytest.raises(ValueError, match="exact 0-to-analysis-max grid"):
        validate_velocity_resolved_overrides(samples, search, [incomplete])

    velocity = tuple(float(value) for value in np.arange(0.0, 2.01, 0.25))
    complete = VelocityResolvedCaptureOverride(
        0,
        0,
        velocity,
        tuple(value == 0.5 for value in velocity),
    )
    with pytest.raises(ValueError, match="no direct evidence"):
        calculate_clustered_cross_section_with_overrides(
            samples,
            search,
            [complete],
            np.asarray([0.0, 0.1, 0.2]),
        )
