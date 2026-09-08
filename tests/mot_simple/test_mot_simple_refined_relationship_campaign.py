"""Focused checks for the refined effective two-level MOT loading campaign."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

import pmot.mot_simple.refined_relationship_campaign as campaign
from pmot.capture_statistics import CaptureVelocitySample, TrajectoryClassification
from pmot.mot_simple.power_loading_study import geometry_csv_text, geometry_rows
from pmot.mot_simple.timeout_audit import AdaptiveAuditEvidence, TimeoutAuditOutcome


def _sample(point, capture_velocity: float = 5.0, *, upper: str = "escaped"):
    return CaptureVelocitySample(
        disc_index=point.disc_index,
        point_index=point.point_index,
        theta_rad=point.theta_rad,
        phi_rad=point.phi_rad,
        theta_prime_rad=point.theta_prime_rad,
        s_m=point.s_m,
        radial_distance_m=point.radial_distance_m,
        initial_position_m=point.initial_position_m,
        incident_unit_vector=point.incident_unit_vector,
        capture_velocity_m_per_s=capture_velocity,
        velocity_resolution_m_per_s=0.25,
        trapped_velocity_lower_m_per_s=capture_velocity,
        untrapped_velocity_upper_m_per_s=capture_velocity + 0.25,
        lower_classification="bounded_core_residence",
        upper_classification=upper,
        lower_entered_trap_core=True,
        upper_entered_trap_core=False,
        lower_core_entry_count=1,
        upper_core_entry_count=0,
    )


def _classification(reason: str, *, elapsed_time_s: float = 0.01):
    trapped = reason in campaign.TRAPPED_TERMINATION_REASONS
    final_radius_m = 1.0e-3 if trapped else 31.0e-3
    return TrajectoryClassification(
        trapped=trapped,
        termination_reason=reason,
        entered_trap_core=trapped,
        core_entry_count=2 if reason == "two_core_entries" else int(trapped),
        elapsed_time_s=elapsed_time_s,
        minimum_radius_m=1.0e-3 if trapped else 4.0e-3,
        final_radius_m=final_radius_m,
        final_position_m=(final_radius_m, 0.0, 0.0),
        final_velocity_m_per_s=(0.0, 0.0, 0.0),
    )


def test_full_audit_classification_rejects_false_core_and_nonfinite_evidence() -> None:
    key = (3, 7)
    payload = campaign._classification_payload(_classification("two_core_entries"))
    payload["core_entry_count"] = 0
    payload["entered_trap_core"] = False
    with pytest.raises(ValueError, match="lacks two core entries"):
        campaign._audit_classification_state(
            payload,
            key=key,
            field="test",
            require_terminal_details=True,
        )

    payload = campaign._classification_payload(_classification("two_core_entries"))
    payload["final_velocity_m_per_s"][1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite terminal state"):
        campaign._audit_classification_state(
            payload,
            key=key,
            field="test",
            require_terminal_details=True,
        )


def _instrumented_evaluations(sample, timeout_speeds=()):
    """Minimal complete-search ledger containing both saved endpoints."""

    results = {
        round(sample.trapped_velocity_lower_m_per_s, 12): _classification(
            sample.lower_classification
        ),
        round(sample.untrapped_velocity_upper_m_per_s, 12): _classification(
            sample.upper_classification
        ),
    }
    for speed in timeout_speeds:
        results[round(float(speed), 12)] = _classification(
            "timeout", elapsed_time_s=0.2
        )
    return tuple((speed, results[speed]) for speed in sorted(results))


def _grid_results(captured):
    return tuple(
        _classification("bounded_core_residence" if value else "escaped")
        for value in captured
    )


def _grid_evidence_json(velocity, captured):
    results = _grid_results(captured)
    return json.dumps(
        [
            {
                "speed_m_per_s": float(speed),
                "coarse_result": campaign._classification_payload(result),
                "fine_result": campaign._classification_payload(result),
            }
            for speed, result in zip(velocity, results, strict=True)
        ],
        separators=(",", ":"),
    )


def _audit(sample, *, timeout: bool = False):
    return {
        "disc_index": sample.disc_index,
        "point_index": sample.point_index,
        "base_timeout_detected": timeout,
        "base_evaluation_timeout_count": int(timeout),
        "base_evaluation_timeout_speeds_m_per_s_json": (
            "[5.25]" if timeout else "[]"
        ),
        "accepted_max_simulation_time_s": 0.2 if timeout else 0.05,
        "accepted_coarse_time_step_s": 5.0e-6,
        "accepted_time_step_s": 2.5e-6 if timeout else 5.0e-6,
        "adaptive_audit_level_count": 0,
        "adaptive_audit_evidence_json": "[]",
        "adaptive_max_duration_s": 0.0,
        "adaptive_coarse_time_step_s": 0.0,
        "adaptive_fine_time_step_s": 0.0,
        "accepted_trapped_velocity_lower_m_per_s": (
            sample.trapped_velocity_lower_m_per_s
        ),
        "accepted_untrapped_velocity_upper_m_per_s": (
            sample.untrapped_velocity_upper_m_per_s
        ),
        "accepted_lower_classification": sample.lower_classification,
        "accepted_upper_classification": sample.upper_classification,
        "base_capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "coarse_capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "fine_capture_velocity_m_per_s": sample.capture_velocity_m_per_s,
        "coarse_trapped_velocity_lower_m_per_s": (
            sample.trapped_velocity_lower_m_per_s
        ),
        "coarse_untrapped_velocity_upper_m_per_s": (
            sample.untrapped_velocity_upper_m_per_s
        ),
        "fine_trapped_velocity_lower_m_per_s": (
            sample.trapped_velocity_lower_m_per_s
        ),
        "fine_untrapped_velocity_upper_m_per_s": (
            sample.untrapped_velocity_upper_m_per_s
        ),
        "coarse_lower_classification": sample.lower_classification,
        "coarse_upper_classification": sample.upper_classification,
        "fine_lower_classification": sample.lower_classification,
        "fine_upper_classification": sample.upper_classification,
        "timeout_node_coarse_results_json": (
            '[{"speed_m_per_s":5.25,"trapped":false,'
            '"termination_reason":"escaped"}]'
            if timeout
            else "[]"
        ),
        "timeout_node_fine_results_json": (
            '[{"speed_m_per_s":5.25,"trapped":false,'
            '"termination_reason":"escaped"}]'
            if timeout
            else "[]"
        ),
        "coarse_evaluation_timeout_count": 0,
        "coarse_evaluation_timeout_speeds_m_per_s_json": "[]",
        "fine_evaluation_timeout_count": 0,
        "fine_evaluation_timeout_speeds_m_per_s_json": "[]",
        "pre_adaptive_coarse_evaluation_timeout_count": 0,
        "pre_adaptive_coarse_evaluation_timeout_speeds_m_per_s_json": "[]",
        "pre_adaptive_fine_evaluation_timeout_count": 0,
        "pre_adaptive_fine_evaluation_timeout_speeds_m_per_s_json": "[]",
        "complete_boundary_audit_evidence_json": "[]",
        "positive_boundary_grid_base_level_index": 0,
        "diagnostic_scalar_boundary_level_index": -1,
        "diagnostic_scalar_boundary_converged": False,
        "diagnostic_scalar_grid_agreement": "not_applicable",
        "timeout_resolution_status": (
            "targeted_timeout_nodes_confirmed_escaped" if timeout else "not_required"
        ),
        "timeout_resolution_reason": "test fixture",
        "zero_threshold_audit_status": "not_applicable",
        "zero_threshold_audit_reason": "test fixture",
        "zero_threshold_grid_evidence_json": "[]",
        "positive_boundary_grid_audit_status": "not_applicable",
        "positive_boundary_grid_audit_reason": "test fixture",
        "positive_boundary_grid_evidence_json": "[]",
        "positive_boundary_grid_duration_s": 0.0,
        "positive_boundary_grid_coarse_time_step_s": 0.0,
        "positive_boundary_grid_fine_time_step_s": 0.0,
        "velocity_resolved_override": False,
    }


def test_requested_grids_and_isolated_default_paths(tmp_path) -> None:
    assert campaign.CAMPAIGN_SCHEMA_VERSION == 6
    assert campaign.POINT_SCHEMA_VERSION == 6
    assert "optimized_v5_20260908" in campaign.CAMPAIGN_NAME
    assert "indeterminate_zero_flux_ray_count" in campaign.AGGREGATE_FIELDNAMES
    assert len(campaign.RAW_SATURATION_VALUES) == 24
    assert campaign.RAW_SATURATION_VALUES[-8:] == (
        60.0,
        70.0,
        80.0,
        90.0,
        100.0,
        110.0,
        120.0,
        125.0,
    )
    assert campaign.EFFECTIVE_SATURATION_VALUES == tuple(
        0.25 * index for index in range(1, 21)
    )
    assert campaign.DETUNING_N_VALUES == tuple(
        -0.25 * index for index in range(2, 25)
    )
    paths = campaign.default_campaign_paths(tmp_path)
    assert paths.statistics == (
        tmp_path / "outputs" / "statistics" / "mot_simple" / campaign.CAMPAIGN_NAME
    )
    assert paths.figures == (
        tmp_path / "outputs" / "figures" / "mot_simple" / campaign.CAMPAIGN_NAME
    )
    assert "mot_multilevel" not in str(paths.statistics)


def test_saturation_conversions_are_exact_and_use_single_beam_center() -> None:
    power_for_one = campaign.saturation_power_w_per_beam(1.0)
    assert power_for_one == pytest.approx(1.0571184782671538e-3)
    assert campaign.on_resonance_saturation_parameter(power_for_one) == pytest.approx(1.0)
    reference_s0 = campaign.on_resonance_saturation_parameter(27.0e-3)
    baseline_n = -15.0e6 / 6.065e6
    assert reference_s0 == pytest.approx(25.54112954704834)
    assert campaign.effective_saturation_from_s0(reference_s0, baseline_n) == pytest.approx(
        1.0029104151628667
    )
    effective_five = campaign.build_relationship_points(
        campaign.EFFECTIVE_STUDY_KEY, [5.0]
    )[0]
    assert effective_five.on_resonance_saturation == pytest.approx(
        5.0 * campaign.detuning_reduction_denominator(baseline_n)
    )
    assert effective_five.cooling_power_w_per_beam == pytest.approx(0.134608234154271)


def test_each_relationship_changes_only_its_named_variable() -> None:
    raw = campaign.build_relationship_points(campaign.RAW_STUDY_KEY, [1.0, 2.0])
    assert raw[0].cooling_detuning_hz == raw[1].cooling_detuning_hz == -15.0e6
    assert raw[0].cooling_power_w_per_beam != raw[1].cooling_power_w_per_beam

    effective = campaign.build_relationship_points(
        campaign.EFFECTIVE_STUDY_KEY, [0.25, 0.5]
    )
    assert effective[0].cooling_detuning_hz == effective[1].cooling_detuning_hz == -15.0e6
    assert effective[0].cooling_power_w_per_beam != effective[1].cooling_power_w_per_beam

    detuning = campaign.build_relationship_points(
        campaign.DETUNING_STUDY_KEY, [-0.5, -0.75]
    )
    assert detuning[0].cooling_power_w_per_beam == detuning[1].cooling_power_w_per_beam == 0.027
    assert detuning[0].on_resonance_saturation == pytest.approx(
        detuning[1].on_resonance_saturation
    )
    assert detuning[0].cooling_detuning_hz != detuning[1].cooling_detuning_hz


def test_common_geometry_exactly_matches_multilevel_comparison_seed() -> None:
    search = campaign.default_search_config()
    discs, points = campaign.generate_common_geometry(search)
    text = geometry_csv_text(geometry_rows(discs, points))
    assert len(discs) == 25
    assert len(points) == 625
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == (
        "02509217f582bc1619712cd31de3fcb34aac11b208c54a4cb228694f36603e17"
    )
    directions = np.asarray([disc.incident_unit_vector for disc in discs])
    assert np.all(np.min(directions, axis=0) < 0.0)
    assert np.all(np.max(directions, axis=0) > 0.0)
    assert all(point.s_m > 0.0 for point in points)
    assert all(point.s_m <= 15.0e-3 for point in points)


def test_point_configuration_synchronizes_power_and_detuning() -> None:
    point = campaign.build_relationship_points(
        campaign.DETUNING_STUDY_KEY, [-3.25]
    )[0]
    apparatus, simple, _, beams = campaign.build_point_configuration(point)
    assert apparatus.cooling.power_w_per_beam == 0.027
    assert apparatus.cooling.detuning_hz == point.cooling_detuning_hz
    assert simple.cooling_detuning_hz == point.cooling_detuning_hz
    assert len(beams) == 6
    assert all(beam.detuning_hz == point.cooling_detuning_hz for beam in beams)
    assert all(beam.intensity_beam.power_w == 0.027 for beam in beams)


def test_point_policy_records_fixed_audit_timesteps_for_nondefault_search() -> None:
    point = campaign.build_relationship_points(campaign.RAW_STUDY_KEY, [1.0])[0]
    search = replace(campaign.default_search_config(), time_step_s=8.0e-6)
    payload = campaign._point_signature_payload(point, search, "geometry-hash")
    policy = payload["endpoint_timeout_policy"]

    assert policy["audit_duration_s"] == pytest.approx(0.2)
    assert policy["audit_coarse_time_step_s"] == pytest.approx(5.0e-6)
    assert policy["fine_time_step_s"] == pytest.approx(2.5e-6)
    assert policy["fine_time_step_s"] != pytest.approx(0.5 * search.time_step_s)
    assert policy["adaptive_node_levels"] == [
        asdict(level) for level in campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS
    ]
    assert policy["complete_positive_boundary_search_levels"] == [
        asdict(level) for level in campaign.COMPLETE_BOUNDARY_SEARCH_LEVELS
    ]
    assert policy["adaptive_node_levels"][-1] == {
        "duration_s": 2.0,
        "coarse_time_step_s": 1.25e-6,
        "fine_time_step_s": 0.625e-6,
    }
    assert policy["adaptive_node_levels"][-2] == {
        "duration_s": 1.0,
        "coarse_time_step_s": 1.25e-6,
        "fine_time_step_s": 0.625e-6,
    }
    assert len(policy["complete_positive_boundary_search_levels"]) == 2
    assert policy["indeterminate_zero_flux_policy"] == {
        "eligible_velocity_m_per_s": 0.0,
        "requires_complete_adaptive_ladder": True,
        "requires_all_positive_speeds_definitive_and_timestep_agreed": True,
        "non_finite_is_admissible": False,
        "capture_state": "indeterminate; neither trapped nor escaped",
        "spectrum_treatment": "omit sigma_capture(0)",
        "loading_treatment": (
            "prepend only the exact weighted-integrand anchor g(0)=0; "
            "do not impute sigma_capture(0)"
        ),
    }


def test_capture_bracket_compatibility_rejects_touching_contradiction() -> None:
    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(
        replace(search, disc_count=1, points_per_disc=1)
    )
    coarse = _sample(points[0], capture_velocity=5.0)
    touching = _sample(points[0], capture_velocity=5.25)
    overlapping = _sample(points[0], capture_velocity=5.125)

    # At 5.25 m/s, coarse says escaped while the touching fine interval says
    # trapped. A shared endpoint therefore cannot count as overlap.
    assert not campaign._compatible_capture_brackets(coarse, touching, search)
    assert campaign._compatible_capture_brackets(coarse, overlapping, search)


def test_campaign_search_stops_near_zero_descent_at_analysis_resolution(
    monkeypatch,
) -> None:
    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(
        replace(search, disc_count=1, points_per_disc=1)
    )
    monkeypatch.setattr(campaign, "_WORKER_BEAMS", [])
    monkeypatch.setattr(campaign, "_WORKER_COIL", object())
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", object())
    evaluated = []

    def fake_classify(_beams, _point, speed, *_args):
        evaluated.append(speed)
        trapped = speed == 0.0
        return SimpleNamespace(
            trapped=trapped,
            termination_reason=("bounded_core_residence" if trapped else "escaped"),
            entered_trap_core=trapped,
            core_entry_count=int(trapped),
        )

    monkeypatch.setattr(campaign, "classify_trajectory", fake_classify)
    sample, timeout_speeds = campaign._evaluate_capture(points[0], search)
    assert timeout_speeds == ()
    assert sample.capture_velocity_m_per_s == 0.0
    assert min(speed for speed in evaluated if speed > 0.0) >= 0.15625
    assert len(evaluated) <= 9


def test_timeout_worker_audits_actual_timeout_speed_without_full_research(monkeypatch) -> None:
    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(replace(search, disc_count=1, points_per_disc=1))
    point = points[0]
    base = _sample(point, upper="timeout")
    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", object())
    monkeypatch.setattr(campaign, "_WORKER_COIL", object())
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", object())
    monkeypatch.setattr(campaign, "_WORKER_BEAMS", [])
    evaluate_calls = []

    def fake_evaluate(*_args):
        evaluate_calls.append(True)
        if len(evaluate_calls) > 1:
            raise AssertionError("definitive targeted evidence must avoid a full re-search")
        return base, (5.25, 5.5)

    def classification(trapped: bool):
        return SimpleNamespace(
            trapped=trapped,
            termination_reason=("bounded_core_residence" if trapped else "escaped"),
            entered_trap_core=trapped,
            core_entry_count=int(trapped),
        )

    monkeypatch.setattr(campaign, "_evaluate_capture", fake_evaluate)
    batch_calls = []

    def fake_batch(_beams, _point, speeds, _coil, _simple, audit_search):
        batch_calls.append((tuple(speeds), audit_search.time_step_s))
        return tuple(classification(speed <= 5.0) for speed in speeds)

    monkeypatch.setattr(campaign, "classify_trajectory_batch", fake_batch)
    result = campaign._capture_worker(point)
    assert result.sample.capture_velocity_m_per_s == 5.0
    assert result.sample.upper_classification == "escaped"
    assert result.audit_row["base_timeout_detected"] is True
    assert result.audit_row["base_evaluation_timeout_count"] == 2
    assert json.loads(
        result.audit_row["base_evaluation_timeout_speeds_m_per_s_json"]
    ) == [5.25, 5.5]
    assert result.audit_row["timeout_resolution_status"] == (
        "targeted_timeout_nodes_confirmed_escaped"
    )
    assert result.audit_row["accepted_max_simulation_time_s"] == 0.2
    assert len(evaluate_calls) == 1
    assert batch_calls == [
        ((5.25, 5.5, 5.0), 5.0e-6),
        ((5.25, 5.5, 5.0), 2.5e-6),
    ]
    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample}, {key: result.audit_row}, {}
    )

    # A success label cannot hide a timestep disagreement at a base-timeout
    # node; the ledger recomputes this from each serialized classification.
    tampered = dict(result.audit_row)
    fine_nodes = json.loads(tampered["timeout_node_fine_results_json"])
    fine_nodes[0].update(
        trapped=True, termination_reason="bounded_core_residence"
    )
    tampered["timeout_node_fine_results_json"] = json.dumps(fine_nodes)
    with pytest.raises(ValueError, match="targeted success"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample}, {key: tampered}, {}
        )


def test_recovered_boundary_accepts_distinct_coarse_and_fine_core_evidence() -> None:
    """The CSV has separate brackets/reasons, but only accepted fine core fields."""

    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(
        replace(search, disc_count=1, points_per_disc=1)
    )
    fine = _sample(points[0], capture_velocity=5.0)
    coarse = replace(
        fine,
        lower_classification="two_core_entries",
        lower_entered_trap_core=True,
        lower_core_entry_count=2,
    )
    coarse_search, fine_search = campaign.audit_searches(search)
    complete_evidence = campaign._complete_boundary_level_payload(
        level_index=0,
        coarse_search=coarse_search,
        fine_search=fine_search,
        coarse_sample=coarse,
        fine_sample=fine,
        coarse_timeout_speeds=(),
        fine_timeout_speeds=(),
        coarse_evaluations=_instrumented_evaluations(coarse),
        fine_evaluations=_instrumented_evaluations(fine),
        production_search=search,
    )
    row = _audit(fine, timeout=True)
    row.update(
        {
            "coarse_lower_classification": coarse.lower_classification,
            "fine_lower_classification": fine.lower_classification,
            "complete_boundary_audit_evidence_json": json.dumps(
                [complete_evidence], separators=(",", ":")
            ),
            "timeout_resolution_status": "dual_step_research_recovered_boundary",
            "timeout_resolution_reason": "synthetic complete 200 ms recovery",
        }
    )
    key = (fine.disc_index, fine.point_index)

    campaign._validate_completed_audit_ledger(
        {key: fine}, {key: row}, {}, search=search
    )


def _configure_positive_boundary_fallback(
    monkeypatch, *, recovery_level, grid_capture_velocity=None
):
    """Install deterministic full-search and exact-grid evidence for one ray."""

    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(
        replace(search, disc_count=1, points_per_disc=1)
    )
    point = points[0]
    base = _sample(point, capture_velocity=5.0, upper="timeout")
    initial = _sample(point, capture_velocity=2.25, upper="timeout")
    level_one = _sample(
        point,
        capture_velocity=2.5,
        upper="escaped" if recovery_level == 1 else "timeout",
    )
    level_two = _sample(
        point,
        capture_velocity=2.75,
        upper="escaped" if recovery_level == 2 else "timeout",
    )
    if grid_capture_velocity is None:
        grid_capture_velocity = {1: 2.5, 2: 2.75}.get(recovery_level, 3.0)
    grid_sample = _sample(point, capture_velocity=grid_capture_velocity)
    evaluate_calls = []
    grid_calls = []

    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", object())
    monkeypatch.setattr(campaign, "_WORKER_COIL", object())
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", object())
    monkeypatch.setattr(campaign, "_WORKER_BEAMS", [])

    def fake_evaluate_instrumented(_point, audit_search):
        assert _point is point
        duration = audit_search.max_simulation_time_s
        time_step = audit_search.time_step_s
        evaluate_calls.append((duration, time_step))
        if np.isclose(duration, search.max_simulation_time_s):
            sample, timeouts = base, (base.untrapped_velocity_upper_m_per_s,)
        elif np.isclose(duration, 0.2):
            sample, timeouts = initial, (initial.untrapped_velocity_upper_m_per_s,)
        elif np.isclose(duration, 0.25):
            sample = level_one
            timeouts = () if recovery_level == 1 else (
                sample.untrapped_velocity_upper_m_per_s,
            )
        elif np.isclose(duration, 0.4):
            sample = level_two
            timeouts = () if recovery_level == 2 else (
                sample.untrapped_velocity_upper_m_per_s,
            )
        else:  # pragma: no cover - catches accidental policy expansion
            raise AssertionError(f"unexpected audit duration {duration}")
        return sample, timeouts, _instrumented_evaluations(sample, timeouts)

    def fake_targeted_batch(_beams, _point, speeds, _coil, _simple, _audit_search):
        assert _point is point
        return tuple(
            _classification(
                "bounded_core_residence"
                if speed <= base.trapped_velocity_lower_m_per_s
                else "timeout"
            )
            for speed in speeds
        )

    def fake_grid_audit(cases, **kwargs):
        assert len(cases) == 1
        case = cases[0]
        grid_calls.append((case, kwargs))
        velocity = tuple(
            float(value)
            for value in np.arange(
                search.analysis_velocity_min_m_per_s,
                search.analysis_velocity_max_m_per_s
                + 0.5 * search.analysis_velocity_step_m_per_s,
                search.analysis_velocity_step_m_per_s,
            )
        )
        captured = tuple(
            speed <= grid_sample.trapped_velocity_lower_m_per_s for speed in velocity
        )
        coarse_grid_results = tuple(
            _classification("bounded_core_residence" if captured_value else "escaped")
            for captured_value in captured
        )
        fine_grid_results = tuple(
            _classification("bounded_core_residence" if captured_value else "escaped")
            for captured_value in captured
        )
        override = campaign.VelocityResolvedCaptureOverride(
            disc_index=case.sample.disc_index,
            point_index=case.sample.point_index,
            velocity_m_per_s=velocity,
            captured=captured,
        )
        return (
            TimeoutAuditOutcome(
                case=case,
                status="velocity_resolved_capture",
                reason="synthetic authoritative complete-grid agreement",
                replacement_sample=grid_sample,
                velocity_override=override,
                coarse_lower_result=_classification(grid_sample.lower_classification),
                fine_lower_result=_classification(grid_sample.lower_classification),
                coarse_upper_result=_classification(grid_sample.upper_classification),
                fine_upper_result=_classification(grid_sample.upper_classification),
                coarse_boundary_sample=grid_sample,
                fine_boundary_sample=grid_sample,
                velocity_grid_m_per_s=velocity,
                coarse_velocity_grid_results=coarse_grid_results,
                fine_velocity_grid_results=fine_grid_results,
            ),
        )

    monkeypatch.setattr(
        campaign, "_evaluate_capture_instrumented", fake_evaluate_instrumented
    )
    monkeypatch.setattr(campaign, "classify_trajectory_batch", fake_targeted_batch)
    monkeypatch.setattr(
        campaign,
        "audit_capture_boundaries_on_velocity_grid_batched",
        fake_grid_audit,
    )
    return point, search, evaluate_calls, grid_calls, grid_sample


def test_positive_boundary_complete_search_recovers_at_250ms(monkeypatch) -> None:
    point, search, evaluate_calls, grid_calls, grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=1)
    )

    result = campaign._capture_worker(point)

    assert evaluate_calls == [
        (search.max_simulation_time_s, search.time_step_s),
        (0.2, 5.0e-6),
        (0.2, 2.5e-6),
        (0.25, 5.0e-6),
        (0.25, 2.5e-6),
    ]
    assert len(grid_calls) == 1
    assert result.sample == grid_sample
    assert result.audit_row["positive_boundary_grid_base_level_index"] == 1
    assert result.audit_row["diagnostic_scalar_boundary_level_index"] == 1
    assert result.audit_row["diagnostic_scalar_boundary_converged"] is True
    assert result.audit_row["diagnostic_scalar_grid_agreement"] == "agree"
    assert result.audit_row["accepted_max_simulation_time_s"] == pytest.approx(0.25)
    assert result.audit_row["accepted_coarse_time_step_s"] == pytest.approx(5.0e-6)
    assert result.audit_row["accepted_time_step_s"] == pytest.approx(2.5e-6)
    assert result.audit_row["timeout_resolution_status"] == (
        "complete_scalar_diagnostics_and_velocity_grid_recovered_capture"
    )


def test_positive_boundary_complete_search_uses_400ms_only_if_needed(
    monkeypatch,
) -> None:
    point, search, evaluate_calls, grid_calls, grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=2)
    )

    result = campaign._capture_worker(point)

    assert evaluate_calls == [
        (search.max_simulation_time_s, search.time_step_s),
        (0.2, 5.0e-6),
        (0.2, 2.5e-6),
        (0.25, 5.0e-6),
        (0.25, 2.5e-6),
        (0.4, 2.5e-6),
        (0.4, 1.25e-6),
    ]
    assert len(grid_calls) == 1
    _, grid_kwargs = grid_calls[0]
    assert grid_kwargs["duration_s"] == pytest.approx(0.4)
    assert tuple(grid_kwargs["adaptive_levels"]) == (
        campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS[2],
        campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS[3],
    )
    assert result.sample == grid_sample
    assert result.audit_row["positive_boundary_grid_base_level_index"] == 2
    assert result.audit_row["diagnostic_scalar_boundary_level_index"] == 2
    assert result.audit_row["diagnostic_scalar_boundary_converged"] is True


def test_positive_boundary_grid_succeeds_when_scalar_never_converges(
    monkeypatch,
) -> None:
    point, search, evaluate_calls, grid_calls, grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=None)
    )

    result = campaign._capture_worker(point)

    assert evaluate_calls == [
        (search.max_simulation_time_s, search.time_step_s),
        (0.2, 5.0e-6),
        (0.2, 2.5e-6),
        (0.25, 5.0e-6),
        (0.25, 2.5e-6),
        (0.4, 2.5e-6),
        (0.4, 1.25e-6),
    ]
    assert len(grid_calls) == 1
    assert result.sample == grid_sample
    assert result.audit_row["positive_boundary_grid_base_level_index"] == 2
    assert result.audit_row["diagnostic_scalar_boundary_level_index"] == -1
    assert result.audit_row["diagnostic_scalar_boundary_converged"] is False
    assert result.audit_row["diagnostic_scalar_grid_agreement"] == "not_converged"
    evidence = json.loads(result.audit_row["complete_boundary_audit_evidence_json"])
    assert [item["level_index"] for item in evidence] == [0, 1, 2]
    assert all(not item["timeout_free"] for item in evidence)
    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample},
        {key: result.audit_row},
        {key: result.velocity_override},
        search=search,
    )


def test_positive_boundary_scalar_grid_disagreement_is_diagnostic(
    monkeypatch,
) -> None:
    point, search, _evaluate_calls, grid_calls, grid_sample = (
        _configure_positive_boundary_fallback(
            monkeypatch, recovery_level=1, grid_capture_velocity=4.0
        )
    )

    result = campaign._capture_worker(point)

    assert len(grid_calls) == 1
    diagnostic_case, _grid_kwargs = grid_calls[0]
    assert diagnostic_case.sample.capture_velocity_m_per_s == pytest.approx(2.5)
    assert result.sample == grid_sample
    assert result.sample.capture_velocity_m_per_s == pytest.approx(4.0)
    assert result.audit_row["diagnostic_scalar_boundary_converged"] is True
    assert result.audit_row["diagnostic_scalar_boundary_level_index"] == 1
    assert result.audit_row["diagnostic_scalar_grid_agreement"] == "disagree"
    assert result.audit_row["fine_capture_velocity_m_per_s"] == pytest.approx(4.0)
    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample},
        {key: result.audit_row},
        {key: result.velocity_override},
        search=search,
    )


def test_positive_boundary_recovery_forces_exact_loading_grid_override(
    monkeypatch,
) -> None:
    point, search, _evaluate_calls, grid_calls, _grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=1)
    )

    result = campaign._capture_worker(point)

    assert len(grid_calls) == 1
    case, grid_kwargs = grid_calls[0]
    assert case.sample.capture_velocity_m_per_s == pytest.approx(2.5)
    assert grid_kwargs["always_override"] is True
    assert grid_kwargs["duration_s"] == pytest.approx(0.25)
    assert grid_kwargs["coarse_time_step_s"] == pytest.approx(5.0e-6)
    assert grid_kwargs["fine_time_step_s"] == pytest.approx(2.5e-6)
    assert tuple(grid_kwargs["adaptive_levels"]) == (
        campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS[1],
        campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS[2],
        campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS[3],
    )
    expected_grid = np.arange(
        search.analysis_velocity_min_m_per_s,
        search.analysis_velocity_max_m_per_s
        + 0.5 * search.analysis_velocity_step_m_per_s,
        search.analysis_velocity_step_m_per_s,
    )
    assert result.velocity_override is not None
    assert np.array_equal(result.velocity_override.velocity_m_per_s, expected_grid)
    assert result.velocity_override.captured[-1] is False
    assert result.audit_row["velocity_resolved_override"] is True
    assert result.audit_row["positive_boundary_grid_audit_status"] == (
        "velocity_resolved_capture"
    )
    grid_evidence = json.loads(
        result.audit_row["positive_boundary_grid_evidence_json"]
    )
    assert len(grid_evidence) == len(expected_grid)
    assert [item["speed_m_per_s"] for item in grid_evidence] == pytest.approx(
        expected_grid
    )


def test_positive_boundary_grid_rejects_arbitrary_nonadaptive_mask_bit_tamper(
    monkeypatch,
) -> None:
    point, search, _evaluate_calls, _grid_calls, _grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=1)
    )
    result = campaign._capture_worker(point)
    assert result.velocity_override is not None
    assert json.loads(result.audit_row["adaptive_audit_evidence_json"]) == []
    tampered_mask = list(result.velocity_override.captured)
    tampered_mask[5] = not tampered_mask[5]
    tampered_override = replace(
        result.velocity_override, captured=tuple(tampered_mask)
    )
    key = (result.sample.disc_index, result.sample.point_index)

    with pytest.raises(ValueError, match="override disagrees with dual-step evidence"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: result.audit_row},
            {key: tampered_override},
            search=search,
        )


def test_positive_boundary_recovery_preserves_every_complete_search_level(
    monkeypatch,
) -> None:
    point, search, _evaluate_calls, _grid_calls, _grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=1)
    )

    result = campaign._capture_worker(point)
    evidence = json.loads(result.audit_row["complete_boundary_audit_evidence_json"])

    assert [item["level_index"] for item in evidence] == [0, 1]
    assert evidence[0]["duration_s"] == pytest.approx(0.2)
    assert evidence[0]["coarse_timeout_speeds_m_per_s"] == [2.5]
    assert evidence[0]["fine_timeout_speeds_m_per_s"] == [2.5]
    assert evidence[0]["timeout_free"] is False
    assert evidence[0]["compatible_definitive_brackets"] is False
    for prefix in ("coarse", "fine"):
        classifications = {
            row["speed_m_per_s"]: row["classification"]["termination_reason"]
            for row in evidence[0][f"{prefix}_evaluations"]
        }
        assert classifications == {
            2.25: "bounded_core_residence",
            2.5: "timeout",
        }
    assert evidence[1]["duration_s"] == pytest.approx(0.25)
    assert evidence[1]["timeout_free"] is True
    assert evidence[1]["compatible_definitive_brackets"] is True
    assert result.audit_row["pre_adaptive_coarse_evaluation_timeout_count"] == 1
    assert result.audit_row["pre_adaptive_fine_evaluation_timeout_count"] == 1

    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample},
        {key: result.audit_row},
        {key: result.velocity_override},
        search=search,
    )


def test_positive_boundary_recovery_rejects_endpoint_evaluation_reason_tamper(
    monkeypatch,
) -> None:
    point, search, _evaluate_calls, _grid_calls, _grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=1)
    )
    result = campaign._capture_worker(point)
    key = (result.sample.disc_index, result.sample.point_index)
    tampered = dict(result.audit_row)
    evidence = json.loads(tampered["complete_boundary_audit_evidence_json"])
    lower_speed = evidence[-1]["fine_sample"][
        "trapped_velocity_lower_m_per_s"
    ]
    lower_evaluation = next(
        item
        for item in evidence[-1]["fine_evaluations"]
        if item["speed_m_per_s"] == lower_speed
    )
    lower_evaluation["classification"]["termination_reason"] = "two_core_entries"
    lower_evaluation["classification"]["core_entry_count"] = 2
    tampered["complete_boundary_audit_evidence_json"] = json.dumps(evidence)

    with pytest.raises(
        ValueError,
        match="endpoint classifications disagree with their evaluations",
    ):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: tampered},
            {key: result.velocity_override},
            search=search,
        )


def test_positive_boundary_recovery_rejects_endpoint_core_evidence_tamper(
    monkeypatch,
) -> None:
    point, search, _evaluate_calls, _grid_calls, _grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=1)
    )
    result = campaign._capture_worker(point)
    key = (result.sample.disc_index, result.sample.point_index)
    tampered = dict(result.audit_row)
    evidence = json.loads(tampered["complete_boundary_audit_evidence_json"])
    evidence[-1]["fine_sample"]["lower_core_entry_count"] = 99
    tampered["complete_boundary_audit_evidence_json"] = json.dumps(evidence)

    with pytest.raises(ValueError, match="core-entry evidence disagrees"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: tampered},
            {key: result.velocity_override},
            search=search,
        )


def test_positive_boundary_recovery_rejects_skipped_first_clean_level(
    monkeypatch,
) -> None:
    point, search, _evaluate_calls, _grid_calls, _grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=2)
    )
    result = campaign._capture_worker(point)
    key = (result.sample.disc_index, result.sample.point_index)
    tampered = dict(result.audit_row)
    evidence = json.loads(tampered["complete_boundary_audit_evidence_json"])
    # Make the 250 ms level independently clean while retaining the claimed
    # 400 ms grid base. A faithful hierarchy must have stopped at 250 ms.
    evidence[1]["coarse_sample"] = evidence[2]["coarse_sample"]
    evidence[1]["fine_sample"] = evidence[2]["fine_sample"]
    evidence[1]["coarse_timeout_speeds_m_per_s"] = []
    evidence[1]["fine_timeout_speeds_m_per_s"] = []
    evidence[1]["coarse_evaluations"] = evidence[2]["coarse_evaluations"]
    evidence[1]["fine_evaluations"] = evidence[2]["fine_evaluations"]
    evidence[1]["compatible_definitive_brackets"] = True
    evidence[1]["timeout_free"] = True
    tampered["complete_boundary_audit_evidence_json"] = json.dumps(evidence)

    with pytest.raises(ValueError, match="first clean scalar-search level"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: tampered},
            {key: result.velocity_override},
            search=search,
        )


def test_ledger_rejects_unused_complete_boundary_evidence(monkeypatch) -> None:
    point, search, _evaluate_calls, _grid_calls, _grid_sample = (
        _configure_positive_boundary_fallback(monkeypatch, recovery_level=1)
    )
    fallback = campaign._capture_worker(point)
    sample = _sample(point)
    row = _audit(sample, timeout=True)
    row["complete_boundary_audit_evidence_json"] = fallback.audit_row[
        "complete_boundary_audit_evidence_json"
    ]
    key = (sample.disc_index, sample.point_index)

    with pytest.raises(ValueError, match="unused complete-boundary evidence"):
        campaign._validate_completed_audit_ledger(
            {key: sample}, {key: row}, {}, search=search
        )


def test_ledger_rejects_tampered_unaudited_integration_fields() -> None:
    search = campaign.default_search_config()
    _discs, points = campaign.generate_common_geometry(search)
    sample = _sample(points[0])
    row = _audit(sample)
    row["accepted_max_simulation_time_s"] = 0.2
    key = (sample.disc_index, sample.point_index)

    with pytest.raises(ValueError, match="unaudited integration field"):
        campaign._validate_completed_audit_ledger(
            {key: sample}, {key: row}, {}, search=search
        )


def test_zero_worker_persists_adaptive_node_evidence(monkeypatch) -> None:
    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(
        replace(search, disc_count=1, points_per_disc=1)
    )
    point = points[0]
    base = replace(
        _sample(point, capture_velocity=0.0, upper="timeout"),
        lower_classification="timeout",
        lower_entered_trap_core=False,
        lower_core_entry_count=0,
    )
    accepted = replace(
        base,
        capture_velocity_m_per_s=0.5,
        trapped_velocity_lower_m_per_s=0.5,
        untrapped_velocity_upper_m_per_s=0.75,
        lower_classification="two_core_entries",
        upper_classification="escaped",
        lower_entered_trap_core=True,
        upper_entered_trap_core=False,
        lower_core_entry_count=2,
        upper_core_entry_count=0,
    )

    def classification(reason: str) -> TrajectoryClassification:
        trapped = reason == "two_core_entries"
        return TrajectoryClassification(
            trapped=trapped,
            termination_reason=reason,
            entered_trap_core=trapped,
            core_entry_count=2 if trapped else 0,
            elapsed_time_s=0.201355,
            minimum_radius_m=0.005,
            final_radius_m=0.03,
            final_position_m=(0.03, 0.0, 0.0),
            final_velocity_m_per_s=(0.8, 0.0, 0.0),
        )

    escaped_coarse = classification("escaped")
    escaped_fine = classification("escaped")
    timeout_coarse = classification("timeout")
    timeout_fine = classification("timeout")
    adaptive_evidence = (
        AdaptiveAuditEvidence(
            level_index=1,
            duration_s=0.25,
            coarse_time_step_s=5.0e-6,
            fine_time_step_s=2.5e-6,
            velocity_m_per_s=0.75,
            coarse_result=timeout_coarse,
            fine_result=timeout_fine,
        ),
        AdaptiveAuditEvidence(
            level_index=2,
            duration_s=0.4,
            coarse_time_step_s=2.5e-6,
            fine_time_step_s=1.25e-6,
            velocity_m_per_s=0.75,
            coarse_result=timeout_coarse,
            fine_result=timeout_fine,
        ),
        AdaptiveAuditEvidence(
            level_index=3,
            duration_s=1.0,
            coarse_time_step_s=1.25e-6,
            fine_time_step_s=0.625e-6,
            velocity_m_per_s=0.75,
            coarse_result=escaped_coarse,
            fine_result=escaped_fine,
        ),
    )
    velocity = tuple(
        float(value)
        for value in np.arange(
            search.analysis_velocity_min_m_per_s,
            search.analysis_velocity_max_m_per_s
            + 0.5 * search.analysis_velocity_step_m_per_s,
            search.analysis_velocity_step_m_per_s,
        )
    )
    captured = tuple(speed <= 0.5 for speed in velocity)
    coarse_grid_results = tuple(
        classification("two_core_entries" if value else "escaped")
        for value in captured
    )
    fine_grid_results = tuple(
        classification("two_core_entries" if value else "escaped")
        for value in captured
    )
    outcome = SimpleNamespace(
        resolved=True,
        loading_admissible=True,
        replacement_sample=accepted,
        coarse_boundary_sample=accepted,
        fine_boundary_sample=accepted,
        status="grid_resolved_capture",
        reason="adaptive test fixture",
        velocity_override=None,
        coarse_lower_result=classification("two_core_entries"),
        fine_lower_result=classification("two_core_entries"),
        coarse_upper_result=escaped_coarse,
        fine_upper_result=escaped_fine,
        adaptive_evidence=adaptive_evidence,
        velocity_grid_m_per_s=velocity,
        coarse_velocity_grid_results=coarse_grid_results,
        fine_velocity_grid_results=fine_grid_results,
    )
    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", object())
    monkeypatch.setattr(campaign, "_WORKER_COIL", object())
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", object())
    monkeypatch.setattr(campaign, "audit_capture_boundary", lambda *args, **kwargs: outcome)

    audit_row = campaign._base_audit_row(point, base, (0.0, 0.25), search)
    result = campaign._apply_zero_threshold_audit(base, audit_row)

    assert result.sample == accepted
    assert result.audit_row["accepted_max_simulation_time_s"] == pytest.approx(0.2)
    assert result.audit_row["accepted_coarse_time_step_s"] == pytest.approx(5.0e-6)
    assert result.audit_row["accepted_time_step_s"] == pytest.approx(2.5e-6)
    assert result.audit_row["adaptive_max_duration_s"] == pytest.approx(1.0)
    assert result.audit_row["adaptive_coarse_time_step_s"] == pytest.approx(1.25e-6)
    assert result.audit_row["adaptive_fine_time_step_s"] == pytest.approx(0.625e-6)
    assert result.audit_row["adaptive_audit_level_count"] == 3
    payload = json.loads(result.audit_row["adaptive_audit_evidence_json"])
    assert [item["level_index"] for item in payload] == [1, 2, 3]
    assert [item["resolved"] for item in payload] == [False, False, True]
    assert payload[-1]["velocity_m_per_s"] == pytest.approx(0.75)
    assert payload[-1]["coarse_result"]["termination_reason"] == "escaped"
    assert payload[-1]["fine_result"]["termination_reason"] == "escaped"
    zero_grid_payload = json.loads(
        result.audit_row["zero_threshold_grid_evidence_json"]
    )
    assert len(zero_grid_payload) == len(velocity)
    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample}, {key: result.audit_row}, {}
    )
    tampered = dict(result.audit_row)
    tampered_payload = json.loads(tampered["adaptive_audit_evidence_json"])
    tampered_payload[-1]["duration_s"] = 0.999
    tampered["adaptive_audit_evidence_json"] = json.dumps(tampered_payload)
    with pytest.raises(ValueError, match="adaptive field duration_s is invalid"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample}, {key: tampered}, {}
        )


def _indeterminate_zero_flux_result(monkeypatch):
    search = replace(
        campaign.default_search_config(),
        disc_count=1,
        points_per_disc=1,
        analysis_velocity_max_m_per_s=1.0,
    )
    _, points = campaign.generate_common_geometry(search)
    point = points[0]
    base = replace(
        _sample(point, capture_velocity=0.0, upper="escaped"),
        lower_classification="timeout",
        lower_entered_trap_core=False,
        lower_core_entry_count=0,
    )
    accepted = replace(
        base,
        lower_classification="indeterminate_zero_flux",
        upper_classification="escaped",
    )
    velocity = tuple(
        float(value)
        for value in np.arange(
            search.analysis_velocity_min_m_per_s,
            search.analysis_velocity_max_m_per_s
            + 0.5 * search.analysis_velocity_step_m_per_s,
            search.analysis_velocity_step_m_per_s,
        )
    )
    escaped = _classification("escaped")
    adaptive_evidence = tuple(
        AdaptiveAuditEvidence(
            level_index=index,
            duration_s=level.duration_s,
            coarse_time_step_s=level.coarse_time_step_s,
            fine_time_step_s=level.fine_time_step_s,
            velocity_m_per_s=0.0,
            coarse_result=_classification(
                "timeout", elapsed_time_s=level.duration_s
            ),
            fine_result=_classification(
                "timeout", elapsed_time_s=level.duration_s
            ),
        )
        for index, level in enumerate(
            campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS, start=1
        )
    )
    final_timeout = adaptive_evidence[-1].fine_result
    coarse_grid = (final_timeout, *(escaped for _ in velocity[1:]))
    fine_grid = (final_timeout, *(escaped for _ in velocity[1:]))
    override = campaign.VelocityResolvedCaptureOverride(
        disc_index=base.disc_index,
        point_index=base.point_index,
        velocity_m_per_s=velocity,
        captured=(None, *(False for _ in velocity[1:])),
    )
    case = campaign.TimeoutAuditCase(
        sample=base,
        apparatus=object(),
        simple_config=object(),
        coil_config=object(),
        search_config=search,
    )
    outcome = TimeoutAuditOutcome(
        case=case,
        status="velocity_resolved_capture_with_indeterminate_zero_flux",
        reason="sole exact-zero node remains timeout after the full ladder",
        replacement_sample=accepted,
        velocity_override=override,
        coarse_lower_result=final_timeout,
        fine_lower_result=final_timeout,
        coarse_upper_result=escaped,
        fine_upper_result=escaped,
        adaptive_evidence=adaptive_evidence,
        coarse_boundary_sample=accepted,
        fine_boundary_sample=accepted,
        velocity_grid_m_per_s=velocity,
        coarse_velocity_grid_results=coarse_grid,
        fine_velocity_grid_results=fine_grid,
    )
    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", case.apparatus)
    monkeypatch.setattr(campaign, "_WORKER_COIL", case.coil_config)
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", case.simple_config)
    audit_row = campaign._base_audit_row(point, base, (0.0,), search)
    result = campaign._apply_zero_threshold_audit(
        base, audit_row, precomputed_outcome=outcome
    )
    return result, search


def test_zero_worker_and_ledger_accept_only_full_provenance_zero_flux(
    monkeypatch,
) -> None:
    result, search = _indeterminate_zero_flux_result(monkeypatch)
    key = (result.sample.disc_index, result.sample.point_index)

    assert result.sample.lower_classification == "indeterminate_zero_flux"
    assert result.velocity_override is not None
    assert result.velocity_override.captured[0] is None
    assert result.audit_row["zero_threshold_audit_status"] == (
        "velocity_resolved_capture_with_indeterminate_zero_flux"
    )
    assert result.audit_row["adaptive_audit_level_count"] == len(
        campaign.DEFAULT_ADAPTIVE_AUDIT_LEVELS
    )
    assert result.audit_row["adaptive_max_duration_s"] == pytest.approx(2.0)
    campaign._validate_completed_audit_ledger(
        {key: result.sample},
        {key: result.audit_row},
        {key: result.velocity_override},
        search=search,
    )

    truncated = dict(result.audit_row)
    evidence = json.loads(truncated["adaptive_audit_evidence_json"])
    truncated["adaptive_audit_evidence_json"] = json.dumps(evidence[:-1])
    truncated["adaptive_audit_level_count"] = len(evidence) - 1
    truncated["adaptive_max_duration_s"] = evidence[-2]["duration_s"]
    truncated["adaptive_coarse_time_step_s"] = evidence[-2][
        "coarse_time_step_s"
    ]
    truncated["adaptive_fine_time_step_s"] = evidence[-2]["fine_time_step_s"]
    with pytest.raises(ValueError):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: truncated},
            {key: result.velocity_override},
            search=search,
        )


def test_indeterminate_zero_flux_propagates_through_point_outputs_and_aggregate(
    tmp_path, monkeypatch
) -> None:
    result, search = _indeterminate_zero_flux_result(monkeypatch)
    assert result.velocity_override is not None
    point = campaign.build_relationship_points(campaign.RAW_STUDY_KEY, [1.0])[0]
    paths = campaign.StudyPaths(tmp_path / "statistics", tmp_path / "figures")
    signature_payload = campaign._point_signature_payload(
        point, search, "geometry-hash"
    )
    signature = campaign._signature(signature_payload)

    summary = campaign._analyze_point(
        [result.sample],
        search,
        point,
        paths,
        signature=signature,
        geometry_hash="geometry-hash",
        overrides=[result.velocity_override],
    )

    assert summary["indeterminate_zero_flux_ray_count"] == 1
    assert summary["loading_rate"]["indeterminate_zero_flux_ray_count"] == 1
    assert summary["loading_rate"]["zero_flux_quadrature_anchor_used"] is True
    assert summary["loading_rate"]["zero_speed_cross_section_imputed"] is False
    spectrum = np.genfromtxt(paths.spectrum_csv, delimiter=",", names=True)
    assert np.atleast_1d(spectrum)[0]["velocity_m_per_s"] == pytest.approx(0.25)
    assert not np.any(np.isclose(np.atleast_1d(spectrum)["velocity_m_per_s"], 0.0))
    assert paths.cross_section_png.is_file()
    assert paths.impact_parameter_png.is_file()

    metadata = campaign._new_point_metadata(
        point,
        search,
        signature_payload,
        signature,
        "geometry-hash",
        paths,
        worker_count=1,
        completed_sample_count=1,
    )
    metadata.update(status="completed", elapsed_wall_time_s=1.0)
    campaign._atomic_write_json(paths.metadata_json, metadata)
    aggregate = campaign._point_aggregate_row(point, search, paths, summary)
    assert aggregate["indeterminate_zero_flux_ray_count"] == 1


def _zero_velocity_resolved_result(monkeypatch, *, topology="island"):
    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(
        replace(search, disc_count=1, points_per_disc=1)
    )
    point = points[0]
    base = replace(
        _sample(point, capture_velocity=0.0),
        lower_classification="escaped",
        lower_entered_trap_core=False,
        lower_core_entry_count=0,
    )
    velocity = tuple(
        float(value)
        for value in np.arange(
            search.analysis_velocity_min_m_per_s,
            search.analysis_velocity_max_m_per_s
            + 0.5 * search.analysis_velocity_step_m_per_s,
            search.analysis_velocity_step_m_per_s,
        )
    )
    if topology == "island":
        captured = tuple(1.0 <= speed <= 1.25 for speed in velocity)
        accepted = base
    elif topology == "all_false":
        captured = tuple(False for _speed in velocity)
        accepted = base
    elif topology == "contiguous":
        captured = tuple(speed <= 1.0 for speed in velocity)
        accepted = _sample(point, capture_velocity=1.0)
    else:  # pragma: no cover - test helper contract
        raise ValueError(f"unknown topology {topology!r}")
    coarse_grid_results = _grid_results(captured)
    fine_grid_results = _grid_results(captured)
    override = campaign.VelocityResolvedCaptureOverride(
        disc_index=base.disc_index,
        point_index=base.point_index,
        velocity_m_per_s=velocity,
        captured=captured,
    )
    case = campaign.TimeoutAuditCase(
        sample=base,
        apparatus=object(),
        simple_config=object(),
        coil_config=object(),
        search_config=search,
    )
    outcome = TimeoutAuditOutcome(
        case=case,
        status="velocity_resolved_capture",
        reason=f"synthetic forced {topology} mask",
        replacement_sample=accepted,
        velocity_override=override,
        coarse_lower_result=coarse_grid_results[
            int(round(accepted.trapped_velocity_lower_m_per_s / 0.25))
        ],
        fine_lower_result=fine_grid_results[
            int(round(accepted.trapped_velocity_lower_m_per_s / 0.25))
        ],
        coarse_upper_result=coarse_grid_results[
            int(round(accepted.untrapped_velocity_upper_m_per_s / 0.25))
        ],
        fine_upper_result=fine_grid_results[
            int(round(accepted.untrapped_velocity_upper_m_per_s / 0.25))
        ],
        coarse_boundary_sample=accepted,
        fine_boundary_sample=accepted,
        velocity_grid_m_per_s=velocity,
        coarse_velocity_grid_results=coarse_grid_results,
        fine_velocity_grid_results=fine_grid_results,
    )
    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", case.apparatus)
    monkeypatch.setattr(campaign, "_WORKER_COIL", case.coil_config)
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", case.simple_config)
    audit_row = campaign._base_audit_row(point, base, (), search)
    result = campaign._apply_zero_threshold_audit(
        base, audit_row, precomputed_outcome=outcome
    )
    return result, search


@pytest.mark.parametrize("topology", ["all_false", "contiguous"])
def test_zero_grid_accepts_forced_authoritative_mask_topologies(
    monkeypatch, topology
) -> None:
    result, search = _zero_velocity_resolved_result(
        monkeypatch, topology=topology
    )
    assert result.velocity_override is not None
    assert result.audit_row["zero_threshold_audit_status"] == (
        "velocity_resolved_capture"
    )
    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample},
        {key: result.audit_row},
        {key: result.velocity_override},
        search=search,
    )


def test_zero_grid_rejects_arbitrary_nonadaptive_override_bit_tamper(
    monkeypatch,
) -> None:
    result, search = _zero_velocity_resolved_result(monkeypatch)
    assert result.velocity_override is not None
    assert json.loads(result.audit_row["adaptive_audit_evidence_json"]) == []
    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample},
        {key: result.audit_row},
        {key: result.velocity_override},
        search=search,
    )

    tampered_mask = list(result.velocity_override.captured)
    tampered_mask[20] = not tampered_mask[20]
    tampered_override = replace(
        result.velocity_override, captured=tuple(tampered_mask)
    )
    with pytest.raises(ValueError, match="zero-grid override disagrees"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: result.audit_row},
            {key: tampered_override},
            search=search,
        )

    tampered_origin = dict(result.audit_row)
    tampered_origin["base_capture_velocity_m_per_s"] = 0.25
    with pytest.raises(ValueError, match="nonzero base threshold"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: tampered_origin},
            {key: result.velocity_override},
            search=search,
        )


def test_zero_grid_rejects_timestep_state_tamper(monkeypatch) -> None:
    result, search = _zero_velocity_resolved_result(monkeypatch)
    assert result.velocity_override is not None
    key = (result.sample.disc_index, result.sample.point_index)
    tampered = dict(result.audit_row)
    evidence = json.loads(tampered["zero_threshold_grid_evidence_json"])
    evidence[20]["coarse_result"].update(
        trapped=True,
        termination_reason="bounded_core_residence",
        entered_trap_core=True,
        core_entry_count=1,
    )
    tampered["zero_threshold_grid_evidence_json"] = json.dumps(evidence)

    with pytest.raises(ValueError, match="zero-grid node is unresolved"):
        campaign._validate_completed_audit_ledger(
            {key: result.sample},
            {key: tampered},
            {key: result.velocity_override},
            search=search,
        )


def test_zero_rebisected_scalar_path_retains_empty_grid_evidence(
    monkeypatch,
) -> None:
    search = campaign.default_search_config()
    _, points = campaign.generate_common_geometry(
        replace(search, disc_count=1, points_per_disc=1)
    )
    point = points[0]
    base = replace(
        _sample(point, capture_velocity=0.0),
        lower_classification="escaped",
        lower_entered_trap_core=False,
        lower_core_entry_count=0,
    )
    accepted = _sample(point, capture_velocity=5.0)
    apparatus = object()
    coil = object()
    simple = object()
    evaluate_calls = []

    def fake_evaluate(_point, audit_search):
        assert _point is point
        evaluate_calls.append(
            (audit_search.max_simulation_time_s, audit_search.time_step_s)
        )
        return accepted, ()

    def fake_audit(case, *, rebisector, **_kwargs):
        coarse_search, fine_search = campaign.audit_searches(search)
        coarse = rebisector([], point, coil, simple, coarse_search)
        fine = rebisector([], point, coil, simple, fine_search)
        trapped = _classification("bounded_core_residence")
        escaped = _classification("escaped")
        return TimeoutAuditOutcome(
            case=case,
            status="rebisected",
            reason="synthetic scalar re-bisection",
            replacement_sample=fine,
            velocity_override=None,
            coarse_lower_result=trapped,
            fine_lower_result=trapped,
            coarse_upper_result=escaped,
            fine_upper_result=escaped,
            coarse_rebisected=coarse,
            fine_rebisected=fine,
            coarse_boundary_sample=coarse,
            fine_boundary_sample=fine,
        )

    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", apparatus)
    monkeypatch.setattr(campaign, "_WORKER_COIL", coil)
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", simple)
    monkeypatch.setattr(campaign, "_evaluate_capture", fake_evaluate)
    monkeypatch.setattr(campaign, "audit_capture_boundary", fake_audit)

    audit_row = campaign._base_audit_row(point, base, (), search)
    result = campaign._apply_zero_threshold_audit(base, audit_row)

    assert evaluate_calls == [(0.2, 5.0e-6), (0.2, 2.5e-6)]
    assert result.sample == accepted
    assert result.audit_row["zero_threshold_audit_status"] == "rebisected"
    assert json.loads(result.audit_row["zero_threshold_grid_evidence_json"]) == []
    key = (result.sample.disc_index, result.sample.point_index)
    campaign._validate_completed_audit_ledger(
        {key: result.sample}, {key: result.audit_row}, {}, search=search
    )


def test_capture_worker_batch_groups_only_zero_rays_and_preserves_order(
    monkeypatch,
) -> None:
    search = replace(
        campaign.default_search_config(), disc_count=1, points_per_disc=3
    )
    _, points = campaign.generate_common_geometry(search)
    base_samples = {
        (0, 0): replace(
            _sample(points[0], capture_velocity=0.0),
            lower_classification="escaped",
            lower_entered_trap_core=False,
            lower_core_entry_count=0,
        ),
        (0, 1): _sample(points[1], capture_velocity=4.0),
        (0, 2): replace(
            _sample(points[2], capture_velocity=0.0),
            lower_classification="escaped",
            lower_entered_trap_core=False,
            lower_core_entry_count=0,
        ),
    }

    def escaped() -> TrajectoryClassification:
        return TrajectoryClassification(
            trapped=False,
            termination_reason="escaped",
            entered_trap_core=False,
            core_entry_count=0,
            elapsed_time_s=0.2,
            minimum_radius_m=0.01,
            final_radius_m=0.03,
            final_position_m=(0.03, 0.0, 0.0),
            final_velocity_m_per_s=(1.0, 0.0, 0.0),
        )

    monkeypatch.setattr(campaign, "_WORKER_BEAMS", [])
    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", object())
    monkeypatch.setattr(campaign, "_WORKER_COIL", object())
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", object())
    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    monkeypatch.setattr(
        campaign,
        "_evaluate_capture",
        lambda point, _search: (
            base_samples[(point.disc_index, point.point_index)],
            (),
        ),
    )
    observed_case_keys = []

    def fake_multi_audit(cases):
        observed_case_keys.extend(
            (case.sample.disc_index, case.sample.point_index) for case in cases
        )
        velocity = tuple(
            float(value)
            for value in np.arange(
                search.analysis_velocity_min_m_per_s,
                search.analysis_velocity_max_m_per_s
                + 0.5 * search.analysis_velocity_step_m_per_s,
                search.analysis_velocity_step_m_per_s,
            )
        )
        grid_results = tuple(escaped() for _speed in velocity)
        return tuple(
            TimeoutAuditOutcome(
                case=case,
                status="confirmed_zero_capture",
                reason="synthetic definitive zero grid",
                replacement_sample=case.sample,
                coarse_boundary_sample=case.sample,
                fine_boundary_sample=case.sample,
                velocity_override=None,
                coarse_lower_result=escaped(),
                fine_lower_result=escaped(),
                coarse_upper_result=escaped(),
                fine_upper_result=escaped(),
                velocity_grid_m_per_s=velocity,
                coarse_velocity_grid_results=grid_results,
                fine_velocity_grid_results=grid_results,
            )
            for case in cases
        )

    monkeypatch.setattr(
        campaign, "audit_zero_capture_boundaries_batched", fake_multi_audit
    )
    results = campaign._capture_worker_batch(points)

    assert observed_case_keys == [(0, 0), (0, 2)]
    assert [result.sample for result in results] == [
        base_samples[(0, 0)],
        base_samples[(0, 1)],
        base_samples[(0, 2)],
    ]
    assert results[1].audit_row["zero_threshold_audit_status"] == "not_applicable"
    assert results[0].audit_row["zero_threshold_audit_status"] == (
        "confirmed_zero_capture"
    )


def test_ray_batches_are_stable_and_enforce_worker_batch_bound(monkeypatch) -> None:
    search = replace(
        campaign.default_search_config(), disc_count=1, points_per_disc=11
    )
    _, points = campaign.generate_common_geometry(search)
    batches = campaign._ray_batches(points)
    assert [len(batch) for batch in batches] == [5, 5, 1]
    assert tuple(point for batch in batches for point in batch) == tuple(points)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        campaign._ray_batches(points, batch_size=0)

    monkeypatch.setattr(campaign, "_WORKER_APPARATUS", object())
    monkeypatch.setattr(campaign, "_WORKER_COIL", object())
    monkeypatch.setattr(campaign, "_WORKER_SIMPLE", object())
    monkeypatch.setattr(campaign, "_WORKER_SEARCH", search)
    with pytest.raises(ValueError, match="at most 5 rays"):
        campaign._capture_worker_batch(points[:6])


def test_relationship_point_writes_resumable_complete_products(tmp_path, monkeypatch) -> None:
    search = replace(
        campaign.default_search_config(),
        disc_count=2,
        points_per_disc=2,
        analysis_velocity_max_m_per_s=6.0,
    )
    discs, points = campaign.generate_common_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    relationship_point = campaign.build_relationship_points(
        campaign.DETUNING_STUDY_KEY, [-1.0]
    )[0]
    paths = campaign.StudyPaths(tmp_path / "statistics", tmp_path / "figures")

    def fake_worker(point):
        sample = _sample(point, capture_velocity=4.0 + 0.25 * point.point_index)
        return campaign.CaptureWorkerResult(
            sample, _audit(sample, timeout=point.point_index == 0)
        )

    monkeypatch.setattr(
        campaign,
        "_capture_worker_batch",
        lambda batch: tuple(fake_worker(point) for point in batch),
    )
    summary = campaign.run_relationship_point(
        relationship_point,
        search=search,
        discs=discs,
        points=points,
        geometry_text=geometry_text,
        geometry_hash=geometry_hash,
        paths=paths,
        worker_count=1,
        resume=False,
    )
    assert summary["sample_count"] == 4
    assert summary["unresolved_timeout_count"] == 0
    assert summary["upper_classification_counts"] == {"escaped": 4}
    assert paths.final_samples_csv.is_file()
    assert paths.partial_samples_csv.read_bytes() == paths.final_samples_csv.read_bytes()
    assert paths.spectrum_csv.is_file()
    assert paths.loading_by_disc_csv.is_file()
    assert paths.loading_json.is_file()
    assert paths.cross_section_png.is_file()
    assert campaign._audit_path(paths).is_file()
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["completed_sample_count"] == 4
    assert metadata["automatically_extended_timeout_ray_count"] == 2
    assert metadata["phase_space"] == "full_sphere"
    assert metadata["repumper_included"] is False

    metadata["automatically_extended_timeout_ray_count"] = 999
    metadata["migration"] = {"source_campaign_name": "synthetic-v3"}
    metadata["migration_history"] = [
        {"source_campaign_name": "synthetic-v2"},
        {"source_campaign_name": "synthetic-v3"},
    ]
    campaign._atomic_write_json(paths.metadata_json, metadata)

    resumed = campaign.run_relationship_point(
        relationship_point,
        search=search,
        discs=discs,
        points=points,
        geometry_text=geometry_text,
        geometry_hash=geometry_hash,
        paths=paths,
        worker_count=1,
        resume=True,
        analyze_only=True,
    )
    assert resumed["loading_rate"] == summary["loading_rate"]
    resumed_metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert resumed_metadata["automatically_extended_timeout_ray_count"] == 2
    assert resumed_metadata["migration"] == {
        "source_campaign_name": "synthetic-v3"
    }
    assert resumed_metadata["migration_history"] == [
        {"source_campaign_name": "synthetic-v2"},
        {"source_campaign_name": "synthetic-v3"},
    ]


def test_audit_count_summary_reconstructs_every_counter() -> None:
    point = campaign.build_relationship_points(campaign.RAW_STUDY_KEY, [1.0])[0]
    search = replace(
        campaign.default_search_config(), disc_count=1, points_per_disc=4
    )
    _, points = campaign.generate_common_geometry(search)
    rows = {}
    for index, launch_point in enumerate(points):
        sample = _sample(launch_point)
        row = _audit(sample, timeout=index in {0, 1, 2})
        row["adaptive_audit_level_count"] = 1 if index in {0, 1} else 0
        row["zero_threshold_audit_status"] = (
            "velocity_resolved_capture_with_indeterminate_zero_flux"
            if index == 0
            else "not_applicable"
        )
        row["timeout_resolution_status"] = {
            0: "adaptive_dual_step_research_recovered_boundary",
            1: "complete_scalar_diagnostics_and_velocity_grid_recovered_capture",
        }.get(index, row["timeout_resolution_status"])
        row["positive_boundary_grid_audit_status"] = (
            "velocity_resolved_capture" if index == 1 else "not_applicable"
        )
        row["pre_adaptive_coarse_evaluation_timeout_count"] = int(index == 1)
        rows[(sample.disc_index, sample.point_index)] = row
    override = campaign.VelocityResolvedCaptureOverride(
        disc_index=0,
        point_index=0,
        velocity_m_per_s=(0.0, 0.25),
        captured=(None, False),
    )

    counts = campaign._audit_count_summary(rows, {(0, 0): override})

    assert counts == {
        "automatically_extended_timeout_ray_count": 3,
        "adaptively_extended_velocity_grid_ray_count": 1,
        "adaptively_extended_audit_ray_count": 2,
        "adaptively_extended_positive_boundary_ray_count": 1,
        "complete_longer_boundary_fallback_ray_count": 1,
        "positive_boundary_grid_override_count": 1,
        "pre_adaptive_positive_research_timeout_ray_count": 1,
        "velocity_resolved_override_count": 1,
        "indeterminate_zero_flux_ray_count": 1,
    }


def test_parallel_worker_failure_aborts_pool_before_saving_checkpoint(
    tmp_path, monkeypatch
) -> None:
    search = replace(
        campaign.default_search_config(),
        disc_count=2,
        points_per_disc=3,
        analysis_velocity_max_m_per_s=6.0,
    )
    discs, points = campaign.generate_common_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    relationship_point = campaign.build_relationship_points(
        campaign.DETUNING_STUDY_KEY, [-1.0]
    )[0]
    paths = campaign.StudyPaths(tmp_path / "statistics", tmp_path / "figures")
    events = []

    class FakeProcess:
        def __init__(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append("terminate")
            self.alive = False

        def join(self, timeout=None):
            events.append(("join", timeout))

    class FakeFuture:
        def __init__(self, outcome):
            self.outcome = outcome

        def result(self):
            if isinstance(self.outcome, BaseException):
                events.append("worker_failure")
                raise self.outcome
            return self.outcome

        def cancel(self):
            events.append("cancel")
            return True

    class FakeExecutor:
        instance = None

        def __init__(self, **_kwargs):
            type(self).instance = self
            self.process = FakeProcess()
            self._processes = {123: self.process}
            self.futures = []
            self.shutdown_calls = []

        def submit(self, _worker, batch):
            if any(
                (point.disc_index, point.point_index) == (1, 2)
                for point in batch
            ):
                outcome = RuntimeError("synthetic worker failure")
            else:
                outcome = tuple(
                    campaign.CaptureWorkerResult(
                        (sample := _sample(point)), _audit(sample)
                    )
                    for point in batch
                )
            future = FakeFuture(outcome)
            self.futures.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures):
            events.append(("shutdown", wait, cancel_futures))
            self.shutdown_calls.append((wait, cancel_futures))

    real_save_checkpoint = campaign._save_checkpoint

    def tracked_save_checkpoint(*args, **kwargs):
        events.append("checkpoint")
        return real_save_checkpoint(*args, **kwargs)

    monkeypatch.setattr(campaign, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(campaign, "as_completed", lambda futures: tuple(futures))
    monkeypatch.setattr(campaign, "_save_checkpoint", tracked_save_checkpoint)

    with pytest.raises(RuntimeError, match="synthetic worker failure"):
        campaign.run_relationship_point(
            relationship_point,
            search=search,
            discs=discs,
            points=points,
            geometry_text=geometry_text,
            geometry_hash=geometry_hash,
            paths=paths,
            worker_count=2,
            resume=False,
        )

    executor = FakeExecutor.instance
    assert executor is not None
    assert executor.shutdown_calls == [(False, True)]
    assert events.count("cancel") == len(executor.futures)
    assert events.index("worker_failure") < events.index("terminate")
    assert events.index("terminate") < events.index("checkpoint")
    saved = campaign.load_capture_velocity_samples(paths.partial_samples_csv)
    assert [(item.disc_index, item.point_index) for item in saved] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
    ]
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert metadata["completed_sample_count"] == 5


def test_resume_from_arbitrary_partial_count_crosses_checkpoint_thresholds(
    tmp_path, monkeypatch
) -> None:
    search = replace(
        campaign.default_search_config(),
        disc_count=1,
        points_per_disc=5,
        analysis_velocity_max_m_per_s=6.0,
    )
    discs, points = campaign.generate_common_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    relationship_point = campaign.build_relationship_points(
        campaign.DETUNING_STUDY_KEY, [-1.0]
    )[0]
    paths = campaign.StudyPaths(tmp_path / "statistics", tmp_path / "figures")

    def envelope(point):
        sample = _sample(point)
        return campaign.CaptureWorkerResult(sample, _audit(sample))

    def fail_after_one(batch):
        wrong_key = replace(_sample(batch[1]), disc_index=999)
        return (
            envelope(batch[0]),
            campaign.CaptureWorkerResult(wrong_key, _audit(wrong_key)),
        )

    monkeypatch.setattr(campaign, "_capture_worker_batch", fail_after_one)
    with pytest.raises(KeyError):
        campaign.run_relationship_point(
            relationship_point,
            search=search,
            discs=discs,
            points=points,
            geometry_text=geometry_text,
            geometry_hash=geometry_hash,
            paths=paths,
            worker_count=1,
            resume=False,
        )
    partial = campaign.load_capture_velocity_samples(paths.partial_samples_csv)
    assert [(item.disc_index, item.point_index) for item in partial] == [(0, 0)]

    checkpoint_counts = []
    real_save_checkpoint = campaign._save_checkpoint

    def tracked_save_checkpoint(*args, **kwargs):
        checkpoint_counts.append(len(args[1]))
        return real_save_checkpoint(*args, **kwargs)

    monkeypatch.setattr(campaign, "CHECKPOINT_EVERY", 2)
    monkeypatch.setattr(
        campaign,
        "_capture_worker_batch",
        lambda batch: tuple(envelope(point) for point in batch),
    )
    monkeypatch.setattr(campaign, "_save_checkpoint", tracked_save_checkpoint)
    summary = campaign.run_relationship_point(
        relationship_point,
        search=search,
        discs=discs,
        points=points,
        geometry_text=geometry_text,
        geometry_hash=geometry_hash,
        paths=paths,
        worker_count=1,
        resume=True,
    )

    assert summary["sample_count"] == 5
    assert checkpoint_counts == [2, 4]


def test_relationship_point_persists_and_uses_finite_speed_capture_mask(
    tmp_path, monkeypatch
) -> None:
    search = replace(
        campaign.default_search_config(),
        disc_count=2,
        points_per_disc=2,
        analysis_velocity_max_m_per_s=6.0,
    )
    discs, points = campaign.generate_common_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    relationship_point = campaign.build_relationship_points(
        campaign.RAW_STUDY_KEY, [0.25]
    )[0]
    paths = campaign.StudyPaths(tmp_path / "statistics", tmp_path / "figures")
    velocity = tuple(float(value) for value in np.arange(0.0, 6.0 + 0.125, 0.25))

    def fake_worker(point):
        sample = replace(
            _sample(point, capture_velocity=0.0),
            lower_classification="escaped",
            lower_entered_trap_core=False,
            lower_core_entry_count=0,
        )
        audit = _audit(sample)
        audit["zero_threshold_audit_status"] = "confirmed_zero_capture"
        audit["accepted_max_simulation_time_s"] = 0.2
        audit["accepted_coarse_time_step_s"] = 5.0e-6
        audit["accepted_time_step_s"] = 2.5e-6
        mask = tuple(
            (1.0 <= speed <= 1.25)
            if (point.disc_index, point.point_index) == (0, 0)
            else False
            for speed in velocity
        )
        audit["zero_threshold_grid_evidence_json"] = _grid_evidence_json(
            velocity, mask
        )
        if (point.disc_index, point.point_index) != (0, 0):
            return campaign.CaptureWorkerResult(sample, audit)
        override = campaign.VelocityResolvedCaptureOverride(
            disc_index=0,
            point_index=0,
            velocity_m_per_s=velocity,
            captured=mask,
        )
        audit["velocity_resolved_override"] = True
        audit["zero_threshold_audit_status"] = "velocity_resolved_capture"
        return campaign.CaptureWorkerResult(sample, audit, override)

    monkeypatch.setattr(
        campaign,
        "_capture_worker_batch",
        lambda batch: tuple(fake_worker(point) for point in batch),
    )
    summary = campaign.run_relationship_point(
        relationship_point,
        search=search,
        discs=discs,
        points=points,
        geometry_text=geometry_text,
        geometry_hash=geometry_hash,
        paths=paths,
        worker_count=1,
        resume=False,
    )
    payload = json.loads(campaign._overrides_path(paths).read_text(encoding="utf-8"))
    assert payload["override_count"] == 1
    assert summary["velocity_resolved_override_count"] == 1
    assert summary["loading_rate"]["loading_rate_mean_atoms_per_s"] > 0.0
    spectrum = np.genfromtxt(paths.spectrum_csv, delimiter=",", names=True)
    row = spectrum[np.isclose(spectrum["velocity_m_per_s"], 1.0)][0]
    assert int(row["captured_count"]) == 1
    resumed = campaign.run_relationship_point(
        relationship_point,
        search=search,
        discs=discs,
        points=points,
        geometry_text=geometry_text,
        geometry_hash=geometry_hash,
        paths=paths,
        worker_count=1,
        resume=True,
        analyze_only=True,
    )
    assert resumed["velocity_resolved_override_count"] == 1
    assert resumed["loading_rate"] == summary["loading_rate"]
    assert json.loads(campaign._overrides_path(paths).read_text(encoding="utf-8")) == payload


def test_point_resume_rejects_any_scientific_signature_change(tmp_path, monkeypatch) -> None:
    search = replace(
        campaign.default_search_config(),
        disc_count=1,
        points_per_disc=1,
        analysis_velocity_max_m_per_s=6.0,
    )
    discs, points = campaign.generate_common_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    point_plan = campaign.build_relationship_points(campaign.RAW_STUDY_KEY, [1.0])[0]
    paths = campaign.StudyPaths(tmp_path / "statistics", tmp_path / "figures")

    def fake_worker(point):
        sample = _sample(point)
        return campaign.CaptureWorkerResult(sample, _audit(sample))

    monkeypatch.setattr(
        campaign,
        "_capture_worker_batch",
        lambda batch: tuple(fake_worker(point) for point in batch),
    )
    campaign.run_relationship_point(
        point_plan,
        search=search,
        discs=discs,
        points=points,
        geometry_text=geometry_text,
        geometry_hash=geometry_hash,
        paths=paths,
        worker_count=1,
        resume=False,
    )
    changed = replace(search, max_simulation_time_s=0.051)
    with pytest.raises(ValueError, match="signature mismatch"):
        campaign.run_relationship_point(
            point_plan,
            search=changed,
            discs=discs,
            points=points,
            geometry_text=geometry_text,
            geometry_hash=geometry_hash,
            paths=paths,
            worker_count=1,
            resume=True,
        )


def test_campaign_resume_clears_stale_failure_after_success(
    tmp_path, monkeypatch
) -> None:
    search = campaign.default_search_config()
    discs, points = campaign.generate_common_geometry(search)
    geometry_text = geometry_csv_text(geometry_rows(discs, points))
    geometry_hash = hashlib.sha256(geometry_text.encode("utf-8")).hexdigest()
    paths = campaign.CampaignPaths(
        statistics=tmp_path / "statistics" / campaign.CAMPAIGN_NAME,
        figures=tmp_path / "figures" / campaign.CAMPAIGN_NAME,
    )
    paths.statistics.mkdir(parents=True)
    paths.figures.mkdir(parents=True)
    metadata = campaign._new_campaign_metadata(paths, search, geometry_hash)
    metadata.update(
        {
            "status": "failed",
            "last_error": "RuntimeError: prior resumable failure",
            "migration": {"source_campaign_name": "synthetic-v4"},
            "migration_history": [
                {"source_campaign_name": "synthetic-v3"},
                {"source_campaign_name": "synthetic-v4"},
            ],
            "stage_status": {
                campaign.RAW_STUDY_KEY: "failed",
                campaign.EFFECTIVE_STUDY_KEY: "completed",
                campaign.DETUNING_STUDY_KEY: "completed",
            },
        }
    )
    campaign._atomic_write_text(paths.geometry_csv, geometry_text)
    campaign._atomic_write_json(paths.metadata_json, metadata)
    monkeypatch.setattr(campaign, "run_loading_relationship", lambda *args, **kwargs: [])

    result = campaign.run_campaign(
        paths=paths,
        search=search,
        worker_count=1,
        resume=True,
        selected_studies=(campaign.RAW_STUDY_KEY,),
    )

    saved = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert saved["status"] == "completed"
    assert saved["stage_status"][campaign.RAW_STUDY_KEY] == "completed"
    assert "last_error" not in result
    assert "last_error" not in saved
    assert saved["migration"] == {"source_campaign_name": "synthetic-v4"}
    assert saved["migration_history"] == [
        {"source_campaign_name": "synthetic-v3"},
        {"source_campaign_name": "synthetic-v4"},
    ]


def test_study_metadata_regeneration_preserves_migration_history(
    tmp_path, monkeypatch
) -> None:
    search = campaign.default_search_config()
    paths = campaign.CampaignPaths(
        statistics=tmp_path / "statistics" / campaign.CAMPAIGN_NAME,
        figures=tmp_path / "figures" / campaign.CAMPAIGN_NAME,
    )
    study_metadata_path = paths.study_metadata_json(campaign.RAW_STUDY_KEY)
    study_metadata_path.parent.mkdir(parents=True)
    paths.figures.mkdir(parents=True)
    migration_record = {"source_campaign_name": "synthetic-v4"}
    migration_history = [
        {"source_campaign_name": "synthetic-v3"},
        migration_record,
    ]
    campaign._atomic_write_json(
        study_metadata_path,
        {
            "schema_version": campaign.CAMPAIGN_SCHEMA_VERSION,
            "migration": migration_record,
            "migration_history": migration_history,
        },
    )
    point = campaign.build_relationship_points(campaign.RAW_STUDY_KEY, [1.0])[0]

    def fake_point(*_args, **kwargs):
        return {"geometry_sha256": kwargs["geometry_hash"]}

    monkeypatch.setattr(campaign, "run_relationship_point", fake_point)
    monkeypatch.setattr(
        campaign,
        "_point_aggregate_row",
        lambda *_args, **_kwargs: {
            "point_index": 0,
            "loading_rate_mean_atoms_per_s": 1.0,
            "indeterminate_zero_flux_ray_count": 0,
        },
    )
    monkeypatch.setattr(
        campaign, "plot_loading_relationship", lambda *_args, **_kwargs: None
    )

    campaign.run_loading_relationship(
        campaign.RAW_STUDY_KEY,
        [point],
        search=search,
        campaign_paths=paths,
        worker_count=1,
        resume=True,
    )

    regenerated = json.loads(study_metadata_path.read_text(encoding="utf-8"))
    assert regenerated["migration"] == migration_record
    assert regenerated["migration_history"] == migration_history


def test_malformed_migration_history_fails_closed() -> None:
    with pytest.raises(ValueError, match="migration history"):
        campaign._preserve_migration_provenance(
            {}, {"migration_history": [{"valid": True}, "invalid"]}
        )


def test_relationship_plot_writes_requested_full_range_and_inset(tmp_path, monkeypatch) -> None:
    rows = []
    for index, value in enumerate(campaign.RAW_SATURATION_VALUES):
        rows.append(
            {
                "s0": value,
                "seff": value / 25.0,
                "detuning_n": -2.5,
                "loading_rate_mean_atoms_per_s": (index + 1) * 1.0e6,
                "loading_rate_t95_lower_atoms_per_s": (index + 0.8) * 1.0e6,
                "loading_rate_t95_upper_atoms_per_s": (index + 1.2) * 1.0e6,
                "indeterminate_zero_flux_ray_count": int(index == 2),
            }
        )
    captured = {}
    original_close = campaign.plt.close

    def capture_close(figure):
        captured["figure"] = figure

    monkeypatch.setattr(campaign.plt, "close", capture_close)
    destination = tmp_path / "raw.png"
    try:
        assert campaign.plot_loading_relationship(
            rows, campaign.RAW_STUDY_KEY, destination
        ) == destination
        assert destination.is_file()
        figure = captured["figure"]
        assert len(figure.axes) == 1
        assert len(figure.axes[0].child_axes) == 1
        assert figure.axes[0].get_legend()._loc == 2
        assert any(
            "indeterminate" in text.get_text().lower()
            for text in figure.axes[0].get_legend().get_texts()
        )
        assert figure.axes[0].get_xlim()[1] > 125.0
        assert 125.0 in figure.axes[0].get_xticks()
    finally:
        original_close(captured.get("figure"))
