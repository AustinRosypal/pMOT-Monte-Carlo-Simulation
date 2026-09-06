from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pmot.capture_statistics import CaptureVelocitySample, TrajectoryClassification
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_simple.configuration import (
    default_simple_mot_apparatus,
    default_simple_mot_config,
)
from pmot.mot_simple.sampling import CaptureSearchConfig
from pmot.mot_simple.power_loading_study import (
    calculate_clustered_cross_section as calculate_clustered_cross_section_legacy,
    calculate_disc_clustered_loading as calculate_disc_clustered_loading_legacy,
)
from pmot.mot_simple.timeout_audit import (
    AdaptiveAuditLevel,
    TimeoutAuditCase,
    VelocityResolvedCaptureOverride,
    adaptive_audit_evidence_to_payload,
    audit_capture_boundary,
    audit_capture_boundaries_on_velocity_grid_batched,
    audit_searches,
    audit_zero_capture_boundaries_batched,
    calculate_clustered_cross_section_with_overrides,
    calculate_disc_clustered_loading_with_overrides,
    capture_predicate_with_overrides,
    velocity_overrides_from_payload,
    velocity_overrides_to_payload,
)


def _classification(trapped: bool, reason: str) -> TrajectoryClassification:
    return TrajectoryClassification(
        trapped=trapped,
        termination_reason=reason,
        entered_trap_core=trapped,
        core_entry_count=2 if trapped else 0,
        elapsed_time_s=0.01,
        minimum_radius_m=1.0e-3 if trapped else 5.0e-3,
        final_radius_m=1.0e-3 if trapped else 31.0e-3,
        final_position_m=(0.0, 0.0, 0.0),
        final_velocity_m_per_s=(0.0, 0.0, 0.0),
    )


def _sample(
    *,
    capture: float = 1.0,
    lower_reason: str = "bounded_core_residence",
    upper_reason: str = "timeout",
) -> CaptureVelocitySample:
    upper = capture + 0.2 if capture > 0.0 else 0.25
    return CaptureVelocitySample(
        disc_index=2,
        point_index=3,
        theta_rad=0.3,
        phi_rad=1.2,
        theta_prime_rad=2.0,
        s_m=4.0e-3,
        radial_distance_m=15.0e-3,
        initial_position_m=(15.0e-3, 4.0e-3, 0.0),
        incident_unit_vector=(-1.0, 0.0, 0.0),
        capture_velocity_m_per_s=capture,
        velocity_resolution_m_per_s=upper - capture,
        trapped_velocity_lower_m_per_s=capture,
        untrapped_velocity_upper_m_per_s=upper,
        lower_classification=lower_reason,
        upper_classification=upper_reason,
        lower_entered_trap_core=lower_reason != "escaped",
        upper_entered_trap_core=False,
        lower_core_entry_count=1,
        upper_core_entry_count=0,
    )


def _case(sample: CaptureVelocitySample, **search_changes) -> TimeoutAuditCase:
    search = replace(
        CaptureSearchConfig(),
        analysis_velocity_min_m_per_s=0.0,
        **search_changes,
    )
    return TimeoutAuditCase(
        sample=sample,
        apparatus=default_simple_mot_apparatus(),
        simple_config=default_simple_mot_config(),
        coil_config=default_anti_helmholtz_config(),
        search_config=search,
    )


def _threshold_sample(template: CaptureVelocitySample, threshold: float) -> CaptureVelocitySample:
    return replace(
        template,
        capture_velocity_m_per_s=threshold,
        trapped_velocity_lower_m_per_s=threshold,
        untrapped_velocity_upper_m_per_s=threshold + 0.2,
        velocity_resolution_m_per_s=0.2,
        lower_classification="bounded_core_residence",
        upper_classification="escaped",
    )


def _assert_complete_resolved_grid(outcome, expected_velocity) -> None:
    assert outcome.velocity_grid_m_per_s == expected_velocity
    assert len(outcome.coarse_velocity_grid_results) == len(expected_velocity)
    assert len(outcome.fine_velocity_grid_results) == len(expected_velocity)
    definitive_reasons = {"two_core_entries", "bounded_core_residence", "escaped"}
    for coarse, fine in zip(
        outcome.coarse_velocity_grid_results,
        outcome.fine_velocity_grid_results,
        strict=True,
    ):
        assert coarse.termination_reason in definitive_reasons
        assert fine.termination_reason in definitive_reasons
        assert coarse.trapped == fine.trapped


def test_audit_search_is_longer_and_dual_timestep() -> None:
    production = CaptureSearchConfig()
    coarse, fine = audit_searches(production)

    assert coarse.max_simulation_time_s == pytest.approx(0.2)
    assert coarse.time_step_s == pytest.approx(5.0e-6)
    assert fine.time_step_s == pytest.approx(2.5e-6)
    assert coarse.velocity_tolerance_m_per_s == pytest.approx(0.249)


def test_saved_upper_timeout_can_be_confirmed_escaped() -> None:
    case = _case(_sample())

    def classifier(beams, point, speed, coil, simple, search):
        return (
            _classification(True, "bounded_core_residence")
            if speed <= 1.0
            else _classification(False, "escaped")
        )

    outcome = audit_capture_boundary(case, classifier=classifier)

    assert outcome.status == "confirmed_boundary"
    assert outcome.resolved
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.upper_classification == "escaped"
    assert outcome.coarse_boundary_sample is not None
    assert outcome.fine_boundary_sample == outcome.replacement_sample
    assert outcome.coarse_boundary_sample.capture_velocity_m_per_s == pytest.approx(
        outcome.coarse_boundary_sample.trapped_velocity_lower_m_per_s
    )


def test_later_trapped_upper_endpoint_requires_dual_rebisect() -> None:
    case = _case(_sample())

    def classifier(beams, point, speed, coil, simple, search):
        return _classification(True, "bounded_core_residence")

    def rebisector(beams, point, coil, simple, search):
        threshold = 2.0 if search.time_step_s == pytest.approx(5.0e-6) else 2.05
        return _threshold_sample(case.sample, threshold)

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        rebisector=rebisector,
    )

    assert outcome.status == "rebisected"
    assert outcome.resolved
    assert outcome.replacement_sample.capture_velocity_m_per_s == pytest.approx(2.05)
    assert outcome.coarse_boundary_sample == outcome.coarse_rebisected
    assert outcome.fine_boundary_sample == outcome.fine_rebisected


def test_timestep_dependent_rebisected_boundaries_fail_closed() -> None:
    case = _case(_sample())

    def classifier(beams, point, speed, coil, simple, search):
        return _classification(True, "bounded_core_residence")

    def rebisector(beams, point, coil, simple, search):
        threshold = 1.0 if search.time_step_s > 3.0e-6 else 3.0
        return _threshold_sample(case.sample, threshold)

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        rebisector=rebisector,
    )

    assert outcome.status == "unresolved"
    assert not outcome.resolved
    assert outcome.replacement_sample is None


def test_touching_rebisected_boundaries_fail_closed_at_shared_speed() -> None:
    case = _case(_sample())

    def classifier(beams, point, speed, coil, simple, search):
        return _classification(True, "bounded_core_residence")

    def rebisector(beams, point, coil, simple, search):
        # Coarse says 1.2 m/s escaped; fine says that exact speed trapped.
        threshold = 1.0 if search.time_step_s > 3.0e-6 else 1.2
        return _threshold_sample(case.sample, threshold)

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        rebisector=rebisector,
    )

    assert outcome.status == "unresolved"
    assert not outcome.resolved
    assert outcome.replacement_sample is None
    assert outcome.coarse_rebisected.untrapped_velocity_upper_m_per_s == pytest.approx(
        outcome.fine_rebisected.trapped_velocity_lower_m_per_s
    )


def test_zero_threshold_direct_grid_can_confirm_no_capture() -> None:
    sample = _sample(capture=0.0, lower_reason="timeout", upper_reason="escaped")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    def classifier(beams, point, speed, coil, simple, search):
        return _classification(False, "escaped")

    outcome = audit_capture_boundary(case, classifier=classifier)

    assert outcome.status == "confirmed_zero_capture"
    assert outcome.resolved
    assert outcome.velocity_override is None
    assert outcome.replacement_sample.lower_classification == "escaped"
    assert outcome.coarse_boundary_sample is not None
    assert outcome.fine_boundary_sample == outcome.replacement_sample


def test_zero_threshold_finite_speed_island_is_retained_as_mask() -> None:
    sample = _sample(capture=0.0, lower_reason="escaped", upper_reason="escaped")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.5,
    )

    def classifier(beams, point, speed, coil, simple, search):
        trapped = 0.25 <= speed <= 1.0
        return _classification(
            trapped,
            "bounded_core_residence" if trapped else "escaped",
        )

    outcome = audit_capture_boundary(case, classifier=classifier)

    assert outcome.status == "velocity_resolved_capture"
    assert outcome.resolved
    assert outcome.replacement_sample.capture_velocity_m_per_s == 0.0
    assert outcome.velocity_override is not None
    captured_nodes = [
        speed
        for speed, captured in zip(
            outcome.velocity_override.velocity_m_per_s,
            outcome.velocity_override.captured,
            strict=True,
        )
        if captured
    ]
    assert captured_nodes == [0.25, 0.5, 0.75, 1.0]
    assert outcome.replacement_sample.upper_classification == "escaped"
    assert outcome.replacement_sample.untrapped_velocity_upper_m_per_s == pytest.approx(1.25)


def test_zero_threshold_batch_scan_matches_scalar_mask() -> None:
    sample = _sample(capture=0.0, lower_reason="escaped", upper_reason="escaped")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.5,
    )

    def classifier(beams, point, speed, coil, simple, search):
        trapped = 0.25 <= speed <= 1.0
        return _classification(
            trapped,
            "bounded_core_residence" if trapped else "escaped",
        )

    batch_calls = []

    def batch_classifier(beams, point, speeds, coil, simple, search):
        batch_calls.append((tuple(speeds), search.time_step_s))
        return tuple(
            classifier(beams, point, speed, coil, simple, search) for speed in speeds
        )

    scalar = audit_capture_boundary(case, classifier=classifier)
    batched = audit_capture_boundary(
        case,
        classifier=classifier,
        batch_classifier=batch_classifier,
    )

    assert len(batch_calls) == 2
    assert batched.status == scalar.status == "velocity_resolved_capture"
    assert batched.replacement_sample == scalar.replacement_sample
    assert batched.velocity_override == scalar.velocity_override


def test_batched_zero_audit_recovers_contiguous_boundary_without_rebisect() -> None:
    sample = _sample(capture=0.0, lower_reason="timeout", upper_reason="timeout")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    def classifier(beams, point, speed, coil, simple, search):
        raise AssertionError("successful batched zero audit must not repeat scalar endpoints")

    def batch_classifier(beams, point, speeds, coil, simple, search):
        return tuple(
            _classification(
                speed <= 0.25,
                "bounded_core_residence" if speed <= 0.25 else "escaped",
            )
            for speed in speeds
        )

    def rebisector(*args, **kwargs):
        raise AssertionError("the direct grid already resolves the requested boundary")

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        batch_classifier=batch_classifier,
        rebisector=rebisector,
    )

    assert outcome.status == "grid_resolved_capture"
    assert outcome.resolved
    assert outcome.velocity_override is None
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.capture_velocity_m_per_s == pytest.approx(0.25)
    assert outcome.replacement_sample.untrapped_velocity_upper_m_per_s == pytest.approx(0.5)
    assert outcome.coarse_boundary_sample is not None
    assert outcome.fine_boundary_sample == outcome.replacement_sample


def test_batched_zero_audit_retains_later_island_after_origin_capture() -> None:
    sample = _sample(capture=0.0, lower_reason="timeout", upper_reason="timeout")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    captured = (True, True, False, True, False)

    def classifier(beams, point, speed, coil, simple, search):
        raise AssertionError("successful batched zero audit must not use scalar fallback")

    def batch_classifier(beams, point, speeds, coil, simple, search):
        return tuple(
            _classification(value, "two_core_entries" if value else "escaped")
            for value in captured
        )

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        batch_classifier=batch_classifier,
    )

    assert outcome.status == "velocity_resolved_capture"
    assert outcome.resolved
    assert outcome.replacement_sample.capture_velocity_m_per_s == pytest.approx(0.25)
    assert outcome.velocity_override is not None
    assert outcome.velocity_override.captured == captured


def test_batched_zero_audit_adaptively_extends_only_unresolved_node() -> None:
    sample = _sample(capture=0.0, lower_reason="timeout", upper_reason="timeout")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )
    calls: list[tuple[tuple[float, ...], float, float]] = []

    def classifier(*args, **kwargs):
        raise AssertionError("successful batched audit must not use scalar fallback")

    def batch_classifier(beams, point, speeds, coil, simple, search):
        calls.append((tuple(speeds), search.max_simulation_time_s, search.time_step_s))
        results = []
        for speed in speeds:
            if speed <= 0.5:
                results.append(_classification(True, "two_core_entries"))
            elif speed == pytest.approx(0.75) and search.max_simulation_time_s <= 0.2:
                results.append(_classification(False, "timeout"))
            else:
                results.append(_classification(False, "escaped"))
        return tuple(results)

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        batch_classifier=batch_classifier,
    )

    assert outcome.status == "grid_resolved_capture"
    assert outcome.resolved
    assert outcome.replacement_sample.capture_velocity_m_per_s == pytest.approx(0.5)
    assert outcome.replacement_sample.untrapped_velocity_upper_m_per_s == pytest.approx(0.75)
    assert len(calls) == 4
    assert calls[0][0] == calls[1][0] == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert calls[2][0] == calls[3][0] == (0.75,)
    assert calls[2][1] == calls[3][1] == pytest.approx(0.25)
    assert len(outcome.adaptive_evidence) == 1
    evidence = outcome.adaptive_evidence[0]
    assert evidence.level_index == 1
    assert evidence.velocity_m_per_s == pytest.approx(0.75)
    assert evidence.resolved
    assert evidence.coarse_result.termination_reason == "escaped"
    assert evidence.fine_result.termination_reason == "escaped"
    assert outcome.coarse_boundary_sample is not None
    assert outcome.fine_boundary_sample == outcome.replacement_sample

    payload = adaptive_audit_evidence_to_payload(outcome.adaptive_evidence)
    assert payload[0]["duration_s"] == pytest.approx(0.25)
    assert payload[0]["coarse_time_step_s"] == pytest.approx(5.0e-6)
    assert payload[0]["fine_time_step_s"] == pytest.approx(2.5e-6)
    assert payload[0]["resolved"] is True


def test_batched_zero_audit_uses_second_level_only_when_needed() -> None:
    sample = _sample(capture=0.0, lower_reason="timeout", upper_reason="timeout")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )
    calls: list[tuple[tuple[float, ...], float, float]] = []

    def classifier(*args, **kwargs):
        raise AssertionError("successful batched audit must not use scalar fallback")

    def batch_classifier(beams, point, speeds, coil, simple, search):
        calls.append((tuple(speeds), search.max_simulation_time_s, search.time_step_s))
        return tuple(
            _classification(
                speed <= 0.5,
                "two_core_entries"
                if speed <= 0.5
                else (
                    "timeout"
                    if speed == pytest.approx(0.75)
                    and search.max_simulation_time_s < 0.4
                    else "escaped"
                ),
            )
            for speed in speeds
        )

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        batch_classifier=batch_classifier,
    )

    assert outcome.status == "grid_resolved_capture"
    assert [item.level_index for item in outcome.adaptive_evidence] == [1, 2]
    assert not outcome.adaptive_evidence[0].resolved
    assert outcome.adaptive_evidence[1].resolved
    assert len(calls) == 6
    assert calls[-2][1] == calls[-1][1] == pytest.approx(0.4)
    assert calls[-2][2] == pytest.approx(2.5e-6)
    assert calls[-1][2] == pytest.approx(1.25e-6)


def test_batched_zero_audit_exhausts_hierarchy_fail_closed() -> None:
    sample = _sample(capture=0.0, lower_reason="timeout", upper_reason="timeout")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    def classifier(*args, **kwargs):
        raise AssertionError("batched audit must not use scalar fallback")

    def batch_classifier(beams, point, speeds, coil, simple, search):
        return tuple(
            _classification(
                speed <= 0.5,
                "two_core_entries"
                if speed <= 0.5
                else ("timeout" if speed == pytest.approx(0.75) else "escaped"),
            )
            for speed in speeds
        )

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        batch_classifier=batch_classifier,
        adaptive_levels=(AdaptiveAuditLevel(0.25, 5.0e-6, 2.5e-6),),
    )

    assert outcome.status == "unresolved"
    assert not outcome.resolved
    assert outcome.replacement_sample is None
    assert "after all adaptive levels" in outcome.reason
    assert len(outcome.adaptive_evidence) == 1
    assert not outcome.adaptive_evidence[0].resolved


def test_multi_ray_zero_grid_batch_matches_individual_audits_and_adaptive_evidence() -> None:
    samples = (
        replace(
            _sample(capture=0.0, lower_reason="timeout", upper_reason="timeout"),
            disc_index=0,
            point_index=1,
            initial_position_m=(15.0e-3, 1.0e-3, 0.0),
        ),
        replace(
            _sample(capture=0.0, lower_reason="timeout", upper_reason="timeout"),
            disc_index=0,
            point_index=2,
            initial_position_m=(15.0e-3, 2.0e-3, 0.0),
        ),
    )
    cases = tuple(
        _case(
            sample,
            analysis_velocity_step_m_per_s=0.25,
            analysis_velocity_max_m_per_s=1.25,
        )
        for sample in samples
    )
    grouped_calls: list[tuple[int, float, float]] = []

    def result_for(ray: int, speed: float, duration: float):
        if ray == 1:
            captured = speed <= 0.5
            unresolved = speed == pytest.approx(0.75) and duration <= 0.2
        else:
            captured = 0.25 <= speed <= 0.5
            unresolved = speed == pytest.approx(1.0) and duration < 0.4
        if unresolved:
            return _classification(False, "timeout")
        return _classification(
            captured, "two_core_entries" if captured else "escaped"
        )

    def initial_classifier(beams, positions, velocities, coil, simple, search):
        grouped_calls.append(
            (len(positions), search.max_simulation_time_s, search.time_step_s)
        )
        speeds = np.linalg.norm(np.asarray(velocities), axis=1)
        rays = np.rint(1.0e3 * np.asarray(positions)[:, 1]).astype(int)
        return tuple(
            result_for(int(ray), float(speed), search.max_simulation_time_s)
            for ray, speed in zip(rays, speeds, strict=True)
        )

    def scalar_classifier(*args, **kwargs):
        raise AssertionError("resolved multi-ray audits must not use scalar fallback")

    grouped = audit_zero_capture_boundaries_batched(
        cases,
        initial_conditions_classifier=initial_classifier,
        scalar_classifier=scalar_classifier,
    )

    def individual_batch(beams, point, speeds, coil, simple, search):
        ray = int(round(1.0e3 * point.initial_position_m[1]))
        return tuple(
            result_for(ray, float(speed), search.max_simulation_time_s)
            for speed in speeds
        )

    individual = tuple(
        audit_capture_boundary(
            case,
            classifier=scalar_classifier,
            batch_classifier=individual_batch,
        )
        for case in cases
    )
    assert len(grouped) == len(individual) == 2
    for actual, expected in zip(grouped, individual, strict=True):
        assert actual.status == expected.status
        assert actual.reason == expected.reason
        assert actual.replacement_sample == expected.replacement_sample
        assert actual.velocity_override == expected.velocity_override
        assert actual.adaptive_evidence == expected.adaptive_evidence
        assert actual.coarse_lower_result == expected.coarse_lower_result
        assert actual.fine_upper_result == expected.fine_upper_result

    # Two full 2-ray x 6-speed calls, then only unresolved nodes at each level.
    assert grouped_calls[:2] == [
        (12, pytest.approx(0.2), pytest.approx(5.0e-6)),
        (12, pytest.approx(0.2), pytest.approx(2.5e-6)),
    ]
    assert grouped_calls[2:4] == [
        (2, pytest.approx(0.25), pytest.approx(5.0e-6)),
        (2, pytest.approx(0.25), pytest.approx(2.5e-6)),
    ]
    assert grouped_calls[4:] == [
        (1, pytest.approx(0.4), pytest.approx(2.5e-6)),
        (1, pytest.approx(0.4), pytest.approx(1.25e-6)),
    ]


def test_multi_ray_zero_grid_batch_rejects_mixed_physics_and_duplicate_keys() -> None:
    sample = _sample(capture=0.0, lower_reason="escaped", upper_reason="escaped")
    case = _case(sample, analysis_velocity_max_m_per_s=1.0)
    with pytest.raises(ValueError, match="unique"):
        audit_zero_capture_boundaries_batched((case, case))
    changed = replace(
        case,
        simple_config=replace(case.simple_config, cooling_detuning_hz=-12.0e6),
        sample=replace(sample, point_index=4),
    )
    with pytest.raises(ValueError, match="physical configuration"):
        audit_zero_capture_boundaries_batched((case, changed))


def test_multi_ray_zero_grid_batch_rejects_malformed_classifier_result_count() -> None:
    case = _case(
        _sample(capture=0.0, lower_reason="escaped", upper_reason="escaped"),
        analysis_velocity_max_m_per_s=1.0,
    )

    def malformed(beams, positions, velocities, coil, simple, search):
        return (_classification(False, "escaped"),)

    with pytest.raises(ValueError, match="results for"):
        audit_zero_capture_boundaries_batched(
            (case,), initial_conditions_classifier=malformed
        )


def test_generic_grid_audit_keeps_positive_monotone_mask_when_requested() -> None:
    sample = replace(
        _sample(capture=0.5, upper_reason="escaped"),
        velocity_resolution_m_per_s=0.25,
        untrapped_velocity_upper_m_per_s=0.75,
    )
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )
    calls: list[tuple[int, float, float]] = []

    def initial_classifier(beams, positions, velocities, coil, simple, search):
        speeds = np.linalg.norm(np.asarray(velocities), axis=1)
        calls.append(
            (len(speeds), search.max_simulation_time_s, search.time_step_s)
        )
        return tuple(
            _classification(
                speed <= 0.5,
                "two_core_entries" if speed <= 0.5 else "escaped",
            )
            for speed in speeds
        )

    def scalar_classifier(*args, **kwargs):
        raise AssertionError("grid-aligned endpoints must not use scalar fallback")

    (outcome,) = audit_capture_boundaries_on_velocity_grid_batched(
        (case,),
        always_override=True,
        initial_conditions_classifier=initial_classifier,
        scalar_classifier=scalar_classifier,
    )

    assert outcome.status == "velocity_resolved_capture"
    assert outcome.resolved
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.capture_velocity_m_per_s == pytest.approx(0.5)
    assert outcome.velocity_override is not None
    assert outcome.velocity_override.captured == (True, True, True, False, False)
    assert "explicit policy" in outcome.reason
    _assert_complete_resolved_grid(outcome, (0.0, 0.25, 0.5, 0.75, 1.0))
    assert calls == [
        (5, pytest.approx(0.2), pytest.approx(5.0e-6)),
        (5, pytest.approx(0.2), pytest.approx(2.5e-6)),
    ]


def test_generic_grid_audit_retains_positive_nonmonotone_mask() -> None:
    sample = replace(
        _sample(capture=0.25, upper_reason="escaped"),
        velocity_resolution_m_per_s=0.25,
        untrapped_velocity_upper_m_per_s=0.5,
    )
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )
    captured = (True, True, False, True, False)

    def initial_classifier(beams, positions, velocities, coil, simple, search):
        speeds = np.linalg.norm(np.asarray(velocities), axis=1)
        return tuple(
            _classification(
                captured[int(round(speed / 0.25))],
                (
                    "bounded_core_residence"
                    if captured[int(round(speed / 0.25))]
                    else "escaped"
                ),
            )
            for speed in speeds
        )

    (outcome,) = audit_capture_boundaries_on_velocity_grid_batched(
        (case,), initial_conditions_classifier=initial_classifier
    )

    assert outcome.status == "velocity_resolved_capture"
    assert outcome.resolved
    assert outcome.velocity_override is not None
    assert outcome.velocity_override.captured == captured
    assert "nonmonotone" in outcome.reason
    _assert_complete_resolved_grid(outcome, (0.0, 0.25, 0.5, 0.75, 1.0))


def test_generic_grid_audit_exhausts_adaptive_nodes_fail_closed() -> None:
    sample = replace(
        _sample(capture=0.5, upper_reason="escaped"),
        velocity_resolution_m_per_s=0.25,
        untrapped_velocity_upper_m_per_s=0.75,
    )
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    def initial_classifier(beams, positions, velocities, coil, simple, search):
        speeds = np.linalg.norm(np.asarray(velocities), axis=1)
        return tuple(
            _classification(
                speed <= 0.5,
                (
                    "two_core_entries"
                    if speed <= 0.5
                    else ("timeout" if speed == pytest.approx(0.75) else "escaped")
                ),
            )
            for speed in speeds
        )

    (outcome,) = audit_capture_boundaries_on_velocity_grid_batched(
        (case,), initial_conditions_classifier=initial_classifier
    )

    assert outcome.status == "unresolved"
    assert not outcome.resolved
    assert outcome.replacement_sample is None
    assert outcome.velocity_override is None
    assert "after all adaptive levels" in outcome.reason
    assert [item.level_index for item in outcome.adaptive_evidence] == [1, 2]
    assert all(not item.resolved for item in outcome.adaptive_evidence)
    assert outcome.velocity_grid_m_per_s == (0.0, 0.25, 0.5, 0.75, 1.0)
    unresolved_index = outcome.velocity_grid_m_per_s.index(0.75)
    assert (
        outcome.coarse_velocity_grid_results[unresolved_index].termination_reason
        == "timeout"
    )
    assert (
        outcome.fine_velocity_grid_results[unresolved_index].termination_reason
        == "timeout"
    )


def test_generic_grid_audit_rejects_captured_high_speed_endpoint() -> None:
    sample = replace(
        _sample(capture=0.5, upper_reason="escaped"),
        velocity_resolution_m_per_s=0.25,
        untrapped_velocity_upper_m_per_s=0.75,
    )
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    def initial_classifier(beams, positions, velocities, coil, simple, search):
        return tuple(
            _classification(True, "bounded_core_residence") for _ in velocities
        )

    (outcome,) = audit_capture_boundaries_on_velocity_grid_batched(
        (case,),
        always_override=True,
        initial_conditions_classifier=initial_classifier,
    )

    assert outcome.status == "unresolved"
    assert not outcome.resolved
    assert outcome.replacement_sample is None
    assert outcome.velocity_override is None
    assert "escaped high-speed endpoint" in outcome.reason
    assert outcome.velocity_grid_m_per_s == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert outcome.coarse_velocity_grid_results[-1].trapped
    assert outcome.fine_velocity_grid_results[-1].trapped


def test_generic_grid_audit_can_retain_authoritative_all_false_mask() -> None:
    sample = replace(
        _sample(capture=0.5, upper_reason="escaped"),
        velocity_resolution_m_per_s=0.25,
        untrapped_velocity_upper_m_per_s=0.75,
    )
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    def initial_classifier(beams, positions, velocities, coil, simple, search):
        return tuple(_classification(False, "escaped") for _ in velocities)

    (outcome,) = audit_capture_boundaries_on_velocity_grid_batched(
        (case,),
        always_override=True,
        initial_conditions_classifier=initial_classifier,
    )

    assert outcome.status == "velocity_resolved_capture"
    assert outcome.resolved
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.capture_velocity_m_per_s == 0.0
    assert outcome.velocity_override is not None
    assert outcome.velocity_override.captured == (False, False, False, False, False)
    _assert_complete_resolved_grid(outcome, (0.0, 0.25, 0.5, 0.75, 1.0))
    captured_at = capture_predicate_with_overrides(
        [outcome.replacement_sample],
        outcome.velocity_grid_m_per_s,
        [outcome.velocity_override],
    )
    assert not any(
        captured_at(outcome.replacement_sample, speed)
        for speed in outcome.velocity_grid_m_per_s
    )


def test_generic_grid_audit_rejects_nonzero_velocity_minimum() -> None:
    sample = replace(
        _sample(capture=0.5, upper_reason="escaped"),
        velocity_resolution_m_per_s=0.25,
        untrapped_velocity_upper_m_per_s=0.75,
    )
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )
    case = replace(
        case,
        search_config=replace(
            case.search_config, analysis_velocity_min_m_per_s=0.25
        ),
    )

    def initial_classifier(beams, positions, velocities, coil, simple, search):
        raise AssertionError("an invalid direct grid must fail before classification")

    with pytest.raises(ValueError, match="require a zero velocity minimum"):
        audit_capture_boundaries_on_velocity_grid_batched(
            (case,),
            always_override=True,
            initial_conditions_classifier=initial_classifier,
        )


def test_zero_threshold_malformed_batch_length_fails_closed() -> None:
    sample = _sample(capture=0.0, lower_reason="escaped", upper_reason="escaped")
    case = _case(
        sample,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )

    def classifier(beams, point, speed, coil, simple, search):
        return _classification(False, "escaped")

    def malformed_batch(beams, point, speeds, coil, simple, search):
        return [_classification(False, "escaped")]

    outcome = audit_capture_boundary(
        case,
        classifier=classifier,
        batch_classifier=malformed_batch,
    )

    assert outcome.status == "unresolved"
    assert not outcome.resolved
    assert "failed closed" in outcome.reason
    assert "1 results" in outcome.reason


def test_direct_mask_overrides_zero_scalar_threshold() -> None:
    sample = _sample(capture=0.0, lower_reason="escaped", upper_reason="escaped")
    velocity = (0.0, 0.25, 0.5, 0.75, 1.0)
    override = VelocityResolvedCaptureOverride(
        sample.disc_index,
        sample.point_index,
        velocity,
        (False, False, True, True, False),
    )
    captured = capture_predicate_with_overrides([sample], velocity, [override])

    assert not captured(sample, 0.25)
    assert captured(sample, 0.5)
    assert captured(sample, 0.75)
    assert not captured(sample, 1.0)


def test_velocity_override_json_round_trip_is_exact_and_sorted() -> None:
    velocity = (0.0, 0.25, 0.5, 0.75, 1.0)
    first = VelocityResolvedCaptureOverride(2, 3, velocity, (False, False, True, False, False))
    second = VelocityResolvedCaptureOverride(0, 1, velocity, (False, True, True, False, False))

    payload = velocity_overrides_to_payload([first, second])
    restored = velocity_overrides_from_payload(payload)

    assert restored == [second, first]
    assert payload["representation"] == "direct_boolean_capture_mask"
    assert payload["override_count"] == 2


def test_override_aggregation_changes_spectrum_and_preserves_disc_cluster_math() -> None:
    velocity = (0.0, 0.25, 0.5, 0.75, 1.0)
    zero = _sample(capture=0.0, lower_reason="escaped", upper_reason="escaped")
    samples = [
        replace(zero, disc_index=0, point_index=0),
        replace(_sample(capture=0.75, upper_reason="escaped"), disc_index=0, point_index=1),
        replace(_sample(capture=0.5, upper_reason="escaped"), disc_index=1, point_index=0),
        replace(_sample(capture=1.0, upper_reason="escaped"), disc_index=1, point_index=1),
    ]
    search = replace(
        CaptureSearchConfig(),
        disc_count=2,
        points_per_disc=2,
        disc_radius_m=1.0,
        analysis_velocity_min_m_per_s=0.0,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )
    override = VelocityResolvedCaptureOverride(
        0,
        0,
        velocity,
        (False, False, True, True, False),
    )

    legacy = calculate_clustered_cross_section_with_overrides(
        samples, search, [], velocity
    )
    with_mask = calculate_clustered_cross_section_with_overrides(
        samples, search, [override], velocity
    )
    assert [row["captured_count"] for row in legacy] == [4, 3, 3, 2, 1]
    assert [row["captured_count"] for row in with_mask] == [3, 3, 4, 3, 1]

    by_disc, loading = calculate_disc_clustered_loading_with_overrides(
        samples,
        search,
        with_mask,
        [override],
    )
    disc_mean = sum(float(row["loading_rate_atoms_per_s"]) for row in by_disc) / 2.0
    assert loading["loading_rate_mean_atoms_per_s"] == pytest.approx(disc_mean)
    assert loading["loading_rate_from_mean_spectrum_atoms_per_s"] == pytest.approx(
        disc_mean
    )
    assert loading["disc_count"] == 2


def test_no_override_loading_path_is_exactly_legacy() -> None:
    velocity = (0.0, 0.25, 0.5, 0.75, 1.0)
    samples = [
        replace(_sample(capture=0.5, upper_reason="escaped"), disc_index=0, point_index=0),
        replace(_sample(capture=1.0, upper_reason="escaped"), disc_index=1, point_index=0),
    ]
    search = replace(
        CaptureSearchConfig(),
        disc_count=2,
        points_per_disc=1,
        disc_radius_m=1.0,
        analysis_velocity_min_m_per_s=0.0,
        analysis_velocity_step_m_per_s=0.25,
        analysis_velocity_max_m_per_s=1.0,
    )
    spectrum = calculate_clustered_cross_section_with_overrides(
        samples, search, [], velocity
    )
    legacy_spectrum = calculate_clustered_cross_section_legacy(
        samples, search, velocity_grid_m_per_s=velocity
    )
    first = calculate_disc_clustered_loading_with_overrides(samples, search, spectrum, [])
    second = calculate_disc_clustered_loading_legacy(samples, search, legacy_spectrum)

    assert spectrum == legacy_spectrum
    assert first == second


def test_nonfinite_endpoint_remains_unresolved() -> None:
    case = _case(_sample(upper_reason="non_finite"))

    def classifier(beams, point, speed, coil, simple, search):
        if speed > 1.0:
            return _classification(False, "non_finite")
        return _classification(True, "bounded_core_residence")

    outcome = audit_capture_boundary(case, classifier=classifier)

    assert outcome.status == "unresolved"
    assert not outcome.resolved
