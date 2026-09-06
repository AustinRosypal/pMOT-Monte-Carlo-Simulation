from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from pmot.capture_statistics import CaptureVelocitySample, TrajectoryClassification
from pmot.mot_multilevel import timeout_audit
from pmot.mot_multilevel.power_loading_study import StudyPaths, save_samples_atomic
from pmot.mot_multilevel.rate_capture import RateCaptureSearchConfig
from pmot.mot_multilevel.refined_relationship_campaign import requested_refined_points
from pmot.mot_multilevel.relationship_sweeps import (
    DETUNING_STUDY_KEY,
    RAW_STUDY_KEY,
    CampaignPaths,
)


def _sample(*, upper: str = "timeout") -> CaptureVelocitySample:
    return CaptureVelocitySample(
        disc_index=2,
        point_index=3,
        theta_rad=1.1,
        phi_rad=2.2,
        theta_prime_rad=0.7,
        s_m=8.0e-3,
        radial_distance_m=15.0e-3,
        initial_position_m=(15.0e-3, 2.0e-3, -1.0e-3),
        incident_unit_vector=(-1.0, 0.0, 0.0),
        capture_velocity_m_per_s=0.1,
        velocity_resolution_m_per_s=0.1,
        trapped_velocity_lower_m_per_s=0.1,
        untrapped_velocity_upper_m_per_s=0.2,
        lower_classification="bounded_core_residence",
        upper_classification=upper,
        lower_entered_trap_core=True,
        upper_entered_trap_core=True,
        lower_core_entry_count=1,
        upper_core_entry_count=1,
    )


def _classification(*, trapped: bool, reason: str) -> TrajectoryClassification:
    return TrajectoryClassification(
        trapped=trapped,
        termination_reason=reason,
        entered_trap_core=trapped,
        core_entry_count=1 if trapped else 0,
        elapsed_time_s=0.06,
        minimum_radius_m=1.0e-3,
        final_radius_m=1.5e-3 if trapped else 30.0e-3,
        final_position_m=(0.0, 0.0, 0.0),
        final_velocity_m_per_s=(0.0, 0.0, 0.0),
    )


def _lower_timeout_sample() -> CaptureVelocitySample:
    return replace(
        _sample(upper="escaped"),
        capture_velocity_m_per_s=0.0,
        velocity_resolution_m_per_s=20.0,
        trapped_velocity_lower_m_per_s=0.0,
        untrapped_velocity_upper_m_per_s=20.0,
        lower_classification="timeout",
        lower_entered_trap_core=False,
        lower_core_entry_count=0,
    )


def _case(tmp_path: Path) -> timeout_audit.TimeoutCase:
    point = requested_refined_points()[DETUNING_STUDY_KEY][0]
    paths = StudyPaths(tmp_path / "statistics", tmp_path / "figures")
    paths.statistics.mkdir(parents=True)
    paths.final_samples_csv.write_text("source\n", encoding="utf-8")
    paths.metadata_json.write_text("{}\n", encoding="utf-8")
    search = replace(
        RateCaptureSearchConfig(),
        disc_count=25,
        points_per_disc=25,
        disc_radius_m=15.0e-3,
        phase_space="full_sphere",
    )
    return timeout_audit.TimeoutCase(
        point=point,
        paths=paths,
        sample=_sample(),
        run_metadata={"capture_search_config": asdict(search)},
        source_sample_sha256=timeout_audit._sha256(paths.final_samples_csv),
        source_metadata_sha256=timeout_audit._sha256(paths.metadata_json),
    )


def test_scan_completed_detuning_timeouts_reads_only_completed_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = CampaignPaths(tmp_path / "statistics", tmp_path / "figures")
    first, second = requested_refined_points()[DETUNING_STUDY_KEY][:2]
    for point, status, upper in (
        (first, "completed", "timeout"),
        (second, "running", "timeout"),
    ):
        paths = campaign.point_paths(point)
        paths.statistics.mkdir(parents=True)
        save_samples_atomic(paths.final_samples_csv, [_sample(upper=upper)])
        paths.metadata_json.write_text(
            json.dumps({"status": status, "expected_sample_count": 1}) + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        timeout_audit, "_validate_completed_point_inputs", lambda *args: None
    )

    cases = timeout_audit.scan_completed_detuning_timeouts(campaign)

    assert len(cases) == 1
    assert cases[0].point == first
    assert (cases[0].sample.disc_index, cases[0].sample.point_index) == (2, 3)


def test_raw_saturation_scan_is_independent_of_detuning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "statistics", tmp_path / "figures")
    raw_point = requested_refined_points()[RAW_STUDY_KEY][0]
    paths = campaign.point_paths(raw_point)
    paths.statistics.mkdir(parents=True)
    save_samples_atomic(paths.final_samples_csv, [_sample()])
    paths.metadata_json.write_text(
        json.dumps({"status": "completed", "expected_sample_count": 1}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        timeout_audit, "_validate_completed_point_inputs", lambda *args: None
    )

    raw_cases = timeout_audit.scan_completed_timeouts(campaign, RAW_STUDY_KEY)
    detuning_cases = timeout_audit.scan_completed_timeouts(
        campaign, DETUNING_STUDY_KEY
    )

    assert len(raw_cases) == 1
    assert raw_cases[0].point == raw_point
    assert detuning_cases == []


def test_completed_point_validation_enforces_refined_geometry_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    point = case.point
    wrong_search = replace(
        RateCaptureSearchConfig(**case.run_metadata["capture_search_config"]),
        disc_radius_m=12.0e-3,
    )
    metadata = {
        "status": "completed",
        "capture_search_config": asdict(wrong_search),
        "expected_sample_count": 25 * 25,
        "completed_sample_count": 25 * 25,
        "cooling_power_w_per_beam": point.cooling_power_w_per_beam,
        "repump_power_w_per_beam": 0.1e-3,
        "cooling_detuning_hz": point.cooling_detuning_hz,
        "effective_saturation": {
            "cooling": {
                "beam_center_on_resonance_saturation_parameter": (
                    point.on_resonance_saturation
                ),
                "beam_center_effective_saturation_parameter": (
                    point.effective_saturation
                ),
            }
        },
    }
    monkeypatch.setattr(timeout_audit, "_verify_signed_run", lambda metadata: {})

    with pytest.raises(ValueError, match="disc_radius_m"):
        timeout_audit._validate_completed_point_inputs(
            point,
            case.paths,
            metadata,
            [case.sample] * (25 * 25),
        )


def test_timeout_that_later_traps_is_rebisected_at_both_timesteps(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    classifier_calls: list[tuple[float, float, float]] = []
    rebisector_calls: list[tuple[float, float, float]] = []

    def classifier(point, speed, search, **kwargs):
        classifier_calls.append(
            (speed, search.max_simulation_time_s, search.time_step_s)
        )
        return _classification(trapped=True, reason="bounded_core_residence")

    def rebisector(point, search, **kwargs):
        rebisector_calls.append(
            (
                search.max_simulation_time_s,
                search.time_step_s,
                search.velocity_tolerance_m_per_s,
            )
        )
        return replace(
            case.sample,
            capture_velocity_m_per_s=0.4,
            trapped_velocity_lower_m_per_s=0.4,
            untrapped_velocity_upper_m_per_s=0.55,
            velocity_resolution_m_per_s=0.15,
            upper_classification="escaped",
            upper_entered_trap_core=False,
            upper_core_entry_count=0,
        )

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        rebisector=rebisector,
        physics=(object(), object(), object(), object()),
    )

    assert outcome.status == "rebisected"
    assert outcome.safe_to_apply
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.upper_classification == "escaped"
    assert {call[2] for call in classifier_calls} == {
        timeout_audit.COARSE_TIME_STEP_S,
        timeout_audit.FINE_TIME_STEP_S,
    }
    assert all(call[1] == timeout_audit.AUDIT_DURATION_S for call in classifier_calls)
    assert rebisector_calls == [
        (
            timeout_audit.AUDIT_DURATION_S,
            timeout_audit.COARSE_TIME_STEP_S,
            timeout_audit.AUDIT_VELOCITY_TOLERANCE_M_PER_S,
        ),
        (
            timeout_audit.AUDIT_DURATION_S,
            timeout_audit.FINE_TIME_STEP_S,
            timeout_audit.AUDIT_VELOCITY_TOLERANCE_M_PER_S,
        ),
    ]


def test_timeout_confirmed_escaped_needs_no_rebisection(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def classifier(point, speed, search, **kwargs):
        if speed == case.sample.trapped_velocity_lower_m_per_s:
            return _classification(trapped=True, reason="bounded_core_residence")
        return _classification(trapped=False, reason="escaped")

    def forbidden_rebisector(*args, **kwargs):
        raise AssertionError("rebisection should not run")

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        rebisector=forbidden_rebisector,
        physics=(object(), object(), object(), object()),
    )

    assert outcome.status == "confirmed_escaped"
    assert outcome.safe_to_apply
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.capture_velocity_m_per_s == pytest.approx(0.1)
    assert outcome.replacement_sample.upper_classification == "escaped"


def test_timestep_disagreement_fails_closed(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def classifier(point, speed, search, **kwargs):
        if speed == case.sample.trapped_velocity_lower_m_per_s:
            return _classification(trapped=True, reason="bounded_core_residence")
        if search.time_step_s == timeout_audit.COARSE_TIME_STEP_S:
            return _classification(trapped=True, reason="bounded_core_residence")
        return _classification(trapped=False, reason="escaped")

    def incompatible_rebisector(point, search, **kwargs):
        lower = 0.2 if search.time_step_s == timeout_audit.COARSE_TIME_STEP_S else 0.8
        return replace(
            case.sample,
            capture_velocity_m_per_s=lower,
            trapped_velocity_lower_m_per_s=lower,
            untrapped_velocity_upper_m_per_s=lower + 0.1,
            velocity_resolution_m_per_s=0.1,
            lower_classification="bounded_core_residence",
            upper_classification="escaped",
        )

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        rebisector=incompatible_rebisector,
        physics=(object(), object(), object(), object()),
    )

    assert outcome.status == "unresolved"
    assert not outcome.safe_to_apply
    assert outcome.replacement_sample is None


def test_saved_endpoint_disagreement_can_resolve_by_fresh_dual_rebisection(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    def classifier(point, speed, search, **kwargs):
        if search.time_step_s == timeout_audit.COARSE_TIME_STEP_S:
            return _classification(trapped=True, reason="bounded_core_residence")
        return _classification(trapped=False, reason="escaped")

    def compatible_rebisector(point, search, **kwargs):
        lower = 0.4 if search.time_step_s == timeout_audit.COARSE_TIME_STEP_S else 0.45
        return replace(
            case.sample,
            capture_velocity_m_per_s=lower,
            trapped_velocity_lower_m_per_s=lower,
            untrapped_velocity_upper_m_per_s=lower + 0.1,
            velocity_resolution_m_per_s=0.1,
            lower_classification="bounded_core_residence",
            upper_classification="escaped",
        )

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        rebisector=compatible_rebisector,
        physics=(object(), object(), object(), object()),
        audit_duration_s=0.2,
    )

    assert outcome.status == "rebisected"
    assert outcome.safe_to_apply
    assert outcome.replacement_sample == outcome.fine_rebisected
    assert "fresh coarse/fine" in outcome.reason


def test_nonfinite_saved_endpoint_fails_before_rebisection(tmp_path: Path) -> None:
    case = _case(tmp_path)

    def classifier(point, speed, search, **kwargs):
        if speed == case.sample.untrapped_velocity_upper_m_per_s:
            return _classification(trapped=False, reason="non_finite")
        return _classification(trapped=True, reason="bounded_core_residence")

    def forbidden_rebisector(*args, **kwargs):
        raise AssertionError("non-finite endpoint must fail before rebisection")

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        rebisector=forbidden_rebisector,
        physics=(object(), object(), object(), object()),
    )

    assert outcome.status == "unresolved"
    assert "non-finite" in outcome.reason


def test_custom_timestep_pair_reaches_serial_classifications(tmp_path: Path) -> None:
    case = _case(tmp_path)
    observed_steps: list[float] = []

    def classifier(point, speed, search, **kwargs):
        observed_steps.append(search.time_step_s)
        if speed == case.sample.trapped_velocity_lower_m_per_s:
            return _classification(trapped=True, reason="bounded_core_residence")
        return _classification(trapped=False, reason="escaped")

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        physics=(object(), object(), object(), object()),
        audit_duration_s=0.2,
        coarse_time_step_s=2.5e-6,
        fine_time_step_s=1.25e-6,
    )

    assert outcome.safe_to_apply
    assert set(observed_steps) == {2.5e-6, 1.25e-6}


@pytest.mark.parametrize(
    ("duration_s", "coarse_step_s", "fine_step_s"),
    [
        (0.0, 5.0e-6, 2.5e-6),
        (0.1, 0.0, 2.5e-6),
        (0.1, 5.0e-6, 0.0),
        (0.1, 2.5e-6, 2.5e-6),
        (0.1, 1.25e-6, 2.5e-6),
        (0.1, float("nan"), 1.25e-6),
    ],
)
def test_invalid_audit_duration_or_timestep_pair_is_refused(
    duration_s: float, coarse_step_s: float, fine_step_s: float
) -> None:
    with pytest.raises(ValueError):
        timeout_audit._validate_audit_search_parameters(
            duration_s, coarse_step_s, fine_step_s
        )


def test_lower_timeout_dual_grid_can_confirm_zero_capture(tmp_path: Path) -> None:
    case = replace(
        _case(tmp_path),
        sample=_lower_timeout_sample(),
        endpoint_kind="lower",
    )

    def classifier(point, speed, search, **kwargs):
        return _classification(trapped=False, reason="escaped")

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        physics=(object(), object(), object(), object()),
    )

    assert outcome.status == "confirmed_zero_capture"
    assert outcome.safe_to_apply
    assert len(outcome.velocity_scan_m_per_s) == 121
    assert outcome.velocity_scan_m_per_s[0] == 0.0
    assert outcome.velocity_scan_m_per_s[-1] == 30.0
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.capture_velocity_m_per_s == 0.0
    assert outcome.replacement_sample.lower_classification == "escaped"
    assert outcome.replacement_sample.upper_classification == "escaped"
    assert outcome.replacement_sample.velocity_resolution_m_per_s == pytest.approx(0.25)


def test_lower_timeout_finite_speed_trapped_island_is_velocity_resolved(
    tmp_path: Path,
) -> None:
    case = replace(
        _case(tmp_path),
        sample=_lower_timeout_sample(),
        endpoint_kind="lower",
    )

    def classifier(point, speed, search, **kwargs):
        if speed == 5.0:
            return _classification(trapped=True, reason="bounded_core_residence")
        return _classification(trapped=False, reason="escaped")

    outcome = timeout_audit.audit_timeout_case(
        case,
        classifier=classifier,
        physics=(object(), object(), object(), object()),
    )

    assert outcome.status == "velocity_resolved_capture"
    assert outcome.safe_to_apply
    assert "direct velocity mask" in outcome.reason
    assert outcome.replacement_sample is not None
    assert outcome.replacement_sample.capture_velocity_m_per_s == 0.0
    override = timeout_audit._velocity_override_from_outcome(outcome)
    assert [
        velocity
        for velocity, captured in zip(
            override.velocity_m_per_s, override.captured, strict=True
        )
        if captured
    ] == [5.0]


def test_velocity_resolved_reanalysis_writes_exact_masked_products(
    tmp_path: Path,
) -> None:
    case = replace(
        _case(tmp_path),
        sample=_lower_timeout_sample(),
        endpoint_kind="lower",
    )
    velocities = tuple(0.25 * index for index in range(121))
    scans = tuple(
        _classification(
            trapped=velocity in {0.5, 0.75, 1.0, 1.25},
            reason=(
                "bounded_core_residence"
                if velocity in {0.5, 0.75, 1.0, 1.25}
                else "escaped"
            ),
        )
        for velocity in velocities
    )
    unresolved = timeout_audit.TimeoutOutcome(
        case,
        "unresolved",
        "finite-speed trapped island detected",
        scans[0],
        scans[0],
        scans[80],
        scans[80],
        None,
        None,
        None,
        velocities,
        scans,
        scans,
    )
    replacement, reason = timeout_audit._velocity_resolved_capture_fallback(
        case, velocities, scans, scans
    )
    outcome = replace(
        unresolved,
        status="velocity_resolved_capture",
        reason=reason,
        replacement_sample=replacement,
    )
    case.paths.loading_json.write_text("{}\n", encoding="utf-8")
    search = RateCaptureSearchConfig(**case.run_metadata["capture_search_config"])

    summary = timeout_audit._reanalyze_with_velocity_overrides(
        case,
        [replacement],
        search,
        {"loading_rate": {}},
        [outcome],
        report_path=tmp_path / "audit.json",
        report_sha256="audit-sha",
    )

    with case.paths.spectrum_csv.open(newline="", encoding="utf-8") as stream:
        spectrum = {
            float(row["velocity_m_per_s"]): int(row["captured_count"])
            for row in csv.DictReader(stream)
        }
    loading = json.loads(case.paths.loading_json.read_text(encoding="utf-8"))
    assert spectrum[0.0] == 0
    assert spectrum[0.25] == 0
    assert spectrum[0.5] == 1
    assert spectrum[1.25] == 1
    assert spectrum[1.5] == 0
    assert loading["loading_rate_mean_atoms_per_s"] > 0.0
    assert loading["velocity_resolved_sample_count"] == 1
    assert summary["loading_rate"] == loading
    assert case.paths.cross_section_png.is_file()
    assert case.paths.loading_by_disc_png.is_file()


def test_parallel_audit_uses_spawn_processes_and_preserves_case_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _case(tmp_path / "first")
    second = replace(
        _case(tmp_path / "second"),
        point=requested_refined_points()[DETUNING_STUDY_KEY][1],
        sample=replace(_sample(), point_index=4),
    )
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    escaped = _classification(trapped=False, reason="escaped")

    def outcome(case):
        return timeout_audit.TimeoutOutcome(
            case,
            "confirmed_escaped",
            "test",
            trapped,
            trapped,
            escaped,
            escaped,
            None,
            None,
            replace(case.sample, upper_classification="escaped"),
        )

    class FakeFuture:
        def __init__(self, case):
            self.case = case

        def result(self):
            return outcome(self.case)

    executor_calls: list[dict[str, object]] = []

    class FakeExecutor:
        def __init__(self, **kwargs):
            executor_calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(
            self,
            worker,
            case,
            audit_duration_s,
            coarse_time_step_s,
            fine_time_step_s,
        ):
            assert worker is timeout_audit._audit_case_worker
            assert audit_duration_s == pytest.approx(0.2)
            assert coarse_time_step_s == pytest.approx(2.5e-6)
            assert fine_time_step_s == pytest.approx(1.25e-6)
            return FakeFuture(case)

    monkeypatch.setattr(timeout_audit, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        timeout_audit,
        "as_completed",
        lambda futures: reversed(list(futures)),
    )

    outcomes, errors = timeout_audit._audit_cases(
        [first, second],
        workers=24,
        audit_duration_s=0.2,
        coarse_time_step_s=2.5e-6,
        fine_time_step_s=1.25e-6,
        classifier=timeout_audit.classify_multilevel_loading_trajectory,
        rebisector=timeout_audit.find_multilevel_capture_velocity,
    )

    assert errors == []
    assert [item.case.point.point_index for item in outcomes] == [0, 1]
    assert executor_calls[0]["max_workers"] == 2
    assert executor_calls[0]["mp_context"].get_start_method() == "spawn"


def test_audit_only_writes_separate_report_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "production_stats", tmp_path / "production_figs")
    campaign.statistics.mkdir(parents=True)
    campaign.metadata_json.write_text(
        json.dumps({"status": "running"}) + "\n", encoding="utf-8"
    )
    case = _case(tmp_path / "case")
    escaped = _classification(trapped=False, reason="escaped")
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    replacement = replace(case.sample, upper_classification="escaped")
    outcome = timeout_audit.TimeoutOutcome(
        case,
        "confirmed_escaped",
        "test",
        trapped,
        trapped,
        escaped,
        escaped,
        None,
        None,
        replacement,
    )
    original = case.paths.final_samples_csv.read_bytes()
    monkeypatch.setattr(
        timeout_audit, "scan_completed_timeouts", lambda paths, study_key: [case]
    )
    monkeypatch.setattr(timeout_audit, "audit_timeout_case", lambda *a, **k: outcome)
    monkeypatch.setattr(timeout_audit, "_predicted_loading_changes", lambda values: [])
    audit_directory = tmp_path / "audit"

    report = timeout_audit.run_timeout_audit(
        paths=campaign, audit_directory=audit_directory, apply=False
    )

    assert report["production_modified"] is False
    assert (audit_directory / "timeout_audit_report.json").is_file()
    assert case.paths.final_samples_csv.read_bytes() == original


def test_apply_report_is_written_before_incomplete_campaign_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "production_stats", tmp_path / "production_figs")
    campaign.statistics.mkdir(parents=True)
    campaign.metadata_json.write_text(
        json.dumps({"status": "running"}) + "\n", encoding="utf-8"
    )
    case = _case(tmp_path / "case")
    escaped = _classification(trapped=False, reason="escaped")
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    outcome = timeout_audit.TimeoutOutcome(
        case,
        "confirmed_escaped",
        "test",
        trapped,
        trapped,
        escaped,
        escaped,
        None,
        None,
        replace(case.sample, upper_classification="escaped"),
    )
    monkeypatch.setattr(
        timeout_audit, "scan_completed_timeouts", lambda paths, study_key: [case]
    )
    monkeypatch.setattr(timeout_audit, "audit_timeout_case", lambda *a, **k: outcome)
    monkeypatch.setattr(timeout_audit, "_predicted_loading_changes", lambda values: [])
    audit_directory = tmp_path / "audit"

    with pytest.raises(RuntimeError, match="campaign is incomplete"):
        timeout_audit.run_timeout_audit(
            paths=campaign, audit_directory=audit_directory, apply=True
        )

    report_path = audit_directory / "timeout_audit_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = json.loads(
        (audit_directory / timeout_audit.APPLY_RESULT_NAME).read_text(encoding="utf-8")
    )
    assert report["production_modified"] is False
    assert report["status"] == "apply_preflight_refused"
    assert result["production_modified"] is False
    assert result["status"] == "refused"
    assert "campaign is incomplete" in result["error"]


def test_cli_defaults_to_read_only() -> None:
    args = timeout_audit.build_argument_parser().parse_args([])
    assert args.apply is False
    assert args.study == "detuning"
    assert args.workers == 1
    assert args.duration_ms == pytest.approx(100.0)
    assert args.coarse_step_us == pytest.approx(5.0)
    assert args.fine_step_us == pytest.approx(2.5)
    assert args.prior_report is None


def test_cli_selects_parallel_raw_saturation_audit() -> None:
    args = timeout_audit.build_argument_parser().parse_args(
        [
            "--study",
            "raw-s0",
            "--workers",
            "24",
            "--duration-ms",
            "200",
            "--coarse-step-us",
            "2.5",
            "--fine-step-us",
            "1.25",
            "--prior-report",
            "prior.json",
        ]
    )
    assert timeout_audit.CLI_STUDIES[args.study] == RAW_STUDY_KEY
    assert args.workers == 24
    assert args.duration_ms == pytest.approx(200.0)
    assert args.coarse_step_us == pytest.approx(2.5)
    assert args.fine_step_us == pytest.approx(1.25)
    assert args.prior_report == Path("prior.json")


def test_worker_count_is_capped_before_dispatch() -> None:
    with pytest.raises(ValueError, match="between 1 and 24"):
        timeout_audit._audit_cases(
            [],
            workers=25,
            audit_duration_s=0.1,
            classifier=timeout_audit.classify_multilevel_loading_trajectory,
            rebisector=timeout_audit.find_multilevel_capture_velocity,
        )


def test_signed_physics_source_drift_fails_closed() -> None:
    source = Path(timeout_audit.__file__).resolve()
    recorded = timeout_audit._sha256(source)
    payload = {
        "physics_source_sha256": {source.name: recorded},
        "geometry_sha256": "geometry",
    }
    metadata = {
        "run_signature_payload": payload,
        "run_signature_sha256": timeout_audit.study_signature(payload),
    }

    assert timeout_audit._verify_signed_run(metadata) == {source.name: recorded}

    changed_payload = {
        **payload,
        "physics_source_sha256": {source.name: "0" * 64},
    }
    changed_metadata = {
        "run_signature_payload": changed_payload,
        "run_signature_sha256": timeout_audit.study_signature(changed_payload),
    }
    with pytest.raises(RuntimeError, match="changed since the production run"):
        timeout_audit._verify_signed_run(changed_metadata)


def test_existing_audit_directory_is_refused_without_overwrite(tmp_path: Path) -> None:
    audit_directory = tmp_path / "existing_audit"
    audit_directory.mkdir()
    marker = audit_directory / "evidence.txt"
    marker.write_text("preserve", encoding="utf-8")
    campaign = CampaignPaths(tmp_path / "statistics", tmp_path / "figures")

    with pytest.raises(FileExistsError):
        timeout_audit.run_timeout_audit(
            paths=campaign,
            audit_directory=audit_directory,
            apply=False,
        )

    assert marker.read_text(encoding="utf-8") == "preserve"


def test_zero_original_loading_rate_has_no_nonfinite_relative_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(tmp_path)
    escaped = _classification(trapped=False, reason="escaped")
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    outcome = timeout_audit.TimeoutOutcome(
        case,
        "confirmed_escaped",
        "test",
        trapped,
        trapped,
        escaped,
        escaped,
        None,
        None,
        replace(case.sample, upper_classification="escaped"),
    )
    monkeypatch.setattr(
        timeout_audit, "load_capture_velocity_samples", lambda path: [case.sample]
    )
    monkeypatch.setattr(
        timeout_audit,
        "calculate_clustered_cross_section_with_overrides",
        lambda *args: [],
    )
    loading = iter(
        [
            ([], {"loading_rate_mean_atoms_per_s": 0.0}),
            ([], {"loading_rate_mean_atoms_per_s": 1.0}),
        ]
    )
    monkeypatch.setattr(
        timeout_audit,
        "calculate_disc_clustered_loading_with_overrides",
        lambda *args: next(loading),
    )

    change = timeout_audit._predicted_loading_changes([outcome])[0]

    assert change["relative_change"] is None


def test_apply_failure_restores_every_modified_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "production_stats", tmp_path / "production_figs")
    case = _case(tmp_path / "case")
    save_samples_atomic(case.paths.partial_samples_csv, [case.sample])
    save_samples_atomic(case.paths.final_samples_csv, [case.sample])
    signature_payload = {"geometry_sha256": "geometry"}
    case = replace(
        case,
        run_metadata={
            "capture_search_config": case.run_metadata["capture_search_config"],
            "run_signature_payload": signature_payload,
            "run_signature_sha256": "base-signature",
        },
        source_sample_sha256=timeout_audit._sha256(case.paths.final_samples_csv),
        source_metadata_sha256=timeout_audit._sha256(case.paths.metadata_json),
    )
    escaped = _classification(trapped=False, reason="escaped")
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    outcome = timeout_audit.TimeoutOutcome(
        case,
        "confirmed_escaped",
        "test",
        trapped,
        trapped,
        escaped,
        escaped,
        None,
        None,
        replace(case.sample, upper_classification="escaped"),
    )
    modified = [case.paths.partial_samples_csv, case.paths.final_samples_csv]
    snapshot = timeout_audit._snapshot_files(modified, require_all=True)
    original = {path: path.read_bytes() for path in modified}
    monkeypatch.setattr(
        timeout_audit, "_campaign_completion_issues", lambda paths, study_key: []
    )
    monkeypatch.setattr(
        timeout_audit, "scan_completed_timeouts", lambda paths, study_key: [case]
    )
    monkeypatch.setattr(
        timeout_audit,
        "_modified_product_paths",
        lambda paths, outcomes, study_key: modified,
    )

    def fail_after_sample_writes(*args, **kwargs):
        raise RuntimeError("injected analysis failure")

    monkeypatch.setattr(
        timeout_audit, "analyze_completed_samples", fail_after_sample_writes
    )
    audit_directory = tmp_path / "audit"
    audit_directory.mkdir()
    report_path = audit_directory / timeout_audit.AUDIT_REPORT_NAME
    intent_path = audit_directory / timeout_audit.APPLY_INTENT_NAME
    _write_apply_report(report_path, outcome)
    _write_apply_intent(intent_path, report_path)

    with pytest.raises(timeout_audit.ApplyTransactionError) as caught:
        timeout_audit._apply_corrections(
            campaign,
            [outcome],
            report_path=report_path,
            audit_directory=audit_directory,
            initial_cases=[case],
                initial_snapshot=snapshot,
                intent_path=intent_path,
                study_key=DETUNING_STUDY_KEY,
            )

    assert caught.value.rollback_complete
    assert all(path.read_bytes() == original[path] for path in modified)
    assert (audit_directory / "production_backups").is_dir()


def test_apply_refuses_changed_timeout_case_set_before_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "production_stats", tmp_path / "production_figs")
    case = _case(tmp_path / "case")
    changed_case = replace(case, source_sample_sha256="changed")
    escaped = _classification(trapped=False, reason="escaped")
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    outcome = timeout_audit.TimeoutOutcome(
        case,
        "confirmed_escaped",
        "test",
        trapped,
        trapped,
        escaped,
        escaped,
        None,
        None,
        replace(case.sample, upper_classification="escaped"),
    )
    monkeypatch.setattr(
        timeout_audit, "_campaign_completion_issues", lambda paths, study_key: []
    )
    monkeypatch.setattr(
        timeout_audit,
        "scan_completed_timeouts",
        lambda paths, study_key: [changed_case],
    )

    def forbidden_backup(*args, **kwargs):
        raise AssertionError("backup must not begin after a case-set race")

    monkeypatch.setattr(timeout_audit, "_backup_files", forbidden_backup)

    with pytest.raises(RuntimeError, match="case set changed"):
        timeout_audit._apply_corrections(
            campaign,
            [outcome],
            report_path=tmp_path / "report.json",
            audit_directory=tmp_path / "audit",
            initial_cases=[case],
                initial_snapshot={},
                intent_path=tmp_path / "intent.json",
                study_key=DETUNING_STUDY_KEY,
            )


def test_successful_apply_preserves_base_signature_and_records_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "production_stats", tmp_path / "production_figs")
    case = _case(tmp_path / "case")
    signature_payload = {"geometry_sha256": "geometry"}
    metadata = {
        "capture_search_config": case.run_metadata["capture_search_config"],
        "run_signature_payload": signature_payload,
        "run_signature_sha256": "base-signature",
    }
    case.paths.metadata_json.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    save_samples_atomic(case.paths.partial_samples_csv, [case.sample])
    save_samples_atomic(case.paths.final_samples_csv, [case.sample])
    case.paths.capture_summary_json.write_text("{}\n", encoding="utf-8")
    case.paths.loading_json.write_text("{}\n", encoding="utf-8")
    case = replace(
        case,
        run_metadata=metadata,
        source_sample_sha256=timeout_audit._sha256(case.paths.final_samples_csv),
        source_metadata_sha256=timeout_audit._sha256(case.paths.metadata_json),
    )
    escaped = _classification(trapped=False, reason="escaped")
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    outcome = timeout_audit.TimeoutOutcome(
        case,
        "confirmed_escaped",
        "test",
        trapped,
        trapped,
        escaped,
        escaped,
        None,
        None,
        replace(case.sample, upper_classification="escaped"),
    )
    campaign.statistics.mkdir(parents=True)
    campaign.figures.mkdir(parents=True)
    campaign.metadata_json.write_text("{}\n", encoding="utf-8")
    sweep_metadata = campaign.study_metadata_json(DETUNING_STUDY_KEY)
    sweep_metadata.parent.mkdir(parents=True)
    sweep_metadata.write_text("{}\n", encoding="utf-8")
    aggregate = campaign.aggregate_csv(DETUNING_STUDY_KEY)
    aggregate.write_text("old\n", encoding="utf-8")
    relationship_plot = campaign.relationship_plot(DETUNING_STUDY_KEY)
    relationship_plot.parent.mkdir(parents=True)
    relationship_plot.write_bytes(b"old plot")
    modified = [
        case.paths.partial_samples_csv,
        case.paths.final_samples_csv,
        case.paths.capture_summary_json,
        case.paths.loading_json,
        case.paths.metadata_json,
        campaign.metadata_json,
        sweep_metadata,
        aggregate,
        relationship_plot,
    ]
    snapshot = timeout_audit._snapshot_files(modified, require_all=True)
    monkeypatch.setattr(
        timeout_audit, "_campaign_completion_issues", lambda paths, study_key: []
    )
    monkeypatch.setattr(
        timeout_audit, "scan_completed_timeouts", lambda paths, study_key: [case]
    )
    monkeypatch.setattr(
        timeout_audit,
        "_modified_product_paths",
        lambda paths, outcomes, study_key: modified,
    )

    def fake_analyze(samples, search, paths, **kwargs):
        summary = {"loading_rate": {"loading_rate_mean_atoms_per_s": 1.0}}
        timeout_audit._atomic_write_json(paths.capture_summary_json, summary)
        timeout_audit._atomic_write_json(paths.loading_json, summary["loading_rate"])
        return summary

    def fake_rebuild(paths, study_key):
        timeout_audit._atomic_write_text(aggregate, "point_index\n0\n")
        relationship_plot.write_bytes(b"new plot")
        return [{"point_index": 0}]

    def fake_post_validate(paths, outcomes, study_key):
        assert (
            case.paths.partial_samples_csv.read_bytes()
            == case.paths.final_samples_csv.read_bytes()
        )

    monkeypatch.setattr(timeout_audit, "analyze_completed_samples", fake_analyze)
    monkeypatch.setattr(timeout_audit, "_rebuild_study_aggregate", fake_rebuild)
    monkeypatch.setattr(timeout_audit, "_validate_post_apply", fake_post_validate)
    audit_directory = tmp_path / "audit"
    audit_directory.mkdir()
    report_path = audit_directory / timeout_audit.AUDIT_REPORT_NAME
    intent_path = audit_directory / timeout_audit.APPLY_INTENT_NAME
    _write_apply_report(report_path, outcome)
    _write_apply_intent(intent_path, report_path)

    application = timeout_audit._apply_corrections(
        campaign,
        [outcome],
        report_path=report_path,
        audit_directory=audit_directory,
        initial_cases=[case],
        initial_snapshot=snapshot,
        intent_path=intent_path,
        study_key=DETUNING_STUDY_KEY,
    )

    corrected = timeout_audit.load_capture_velocity_samples(
        case.paths.final_samples_csv
    )[0]
    applied_metadata = json.loads(case.paths.metadata_json.read_text(encoding="utf-8"))
    applied_loading = json.loads(case.paths.loading_json.read_text(encoding="utf-8"))
    applied_summary = json.loads(
        case.paths.capture_summary_json.read_text(encoding="utf-8")
    )
    assert corrected.upper_classification == "escaped"
    assert case.paths.partial_samples_csv.read_bytes() == case.paths.final_samples_csv.read_bytes()
    assert applied_summary["loading_rate"] == applied_loading
    assert applied_metadata["run_signature_sha256"] == "base-signature"
    assert application["dataset_revision_sha256"]
    assert application["pre_apply_sha256"] != application["post_apply_sha256"]
    assert application["status"] == "applied"
    assert json.loads(
        (audit_directory / timeout_audit.APPLY_RESULT_NAME).read_text(
            encoding="utf-8"
        )
    )["dataset_revision_sha256"] == application["dataset_revision_sha256"]


def _resolved_outcome(case: timeout_audit.TimeoutCase) -> timeout_audit.TimeoutOutcome:
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    escaped = _classification(trapped=False, reason="escaped")
    return timeout_audit.TimeoutOutcome(
        case,
        "confirmed_escaped",
        "saved bracket is definitive",
        trapped,
        trapped,
        escaped,
        escaped,
        None,
        None,
        replace(
            case.sample,
            lower_classification="bounded_core_residence",
            upper_classification="escaped",
            lower_entered_trap_core=True,
            upper_entered_trap_core=False,
            lower_core_entry_count=1,
            upper_core_entry_count=0,
        ),
    )


def _unresolved_outcome(case: timeout_audit.TimeoutCase) -> timeout_audit.TimeoutOutcome:
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    timed_out = _classification(trapped=False, reason="timeout")
    return timeout_audit.TimeoutOutcome(
        case,
        "unresolved",
        "still timed out",
        trapped,
        trapped,
        timed_out,
        timed_out,
        None,
        None,
        None,
    )


def _write_prior_report(
    path: Path,
    campaign: CampaignPaths,
    outcomes: list[timeout_audit.TimeoutOutcome],
    *,
    duration_s: float = 0.1,
    coarse_time_step_s: float = timeout_audit.COARSE_TIME_STEP_S,
    fine_time_step_s: float = timeout_audit.FINE_TIME_STEP_S,
    errors: list[dict[str, object]] | None = None,
) -> None:
    resolved_count = sum(outcome.safe_to_apply for outcome in outcomes)
    unresolved_count = sum(not outcome.safe_to_apply for outcome in outcomes) + len(
        errors or []
    )
    timeout_audit._atomic_write_json(
        path,
        {
            "schema_version": timeout_audit.SCHEMA_VERSION,
            "status": "audited",
            "study_key": DETUNING_STUDY_KEY,
            "production_statistics_root": str(campaign.statistics.resolve()),
            "production_figures_root": str(campaign.figures.resolve()),
            "audit_duration_s": duration_s,
            "audit_time_steps_s": [
                coarse_time_step_s,
                fine_time_step_s,
            ],
            "audit_runtime_source_sha256": (
                timeout_audit._audit_runtime_source_hashes()
            ),
            "timeout_case_count": len(outcomes) + len(errors or []),
            "resolved_case_count": resolved_count,
            "unresolved_case_count": unresolved_count,
            "campaign_complete": True,
            "safe_to_apply": bool(unresolved_count == 0),
            "cases": [
                timeout_audit._outcome_payload(
                    outcome,
                    duration_s,
                    coarse_time_step_s,
                    fine_time_step_s,
                )
                for outcome in outcomes
            ],
            "audit_errors": errors or [],
        },
    )


def _write_apply_report(
    path: Path,
    outcome: timeout_audit.TimeoutOutcome,
    *,
    duration_s: float = 0.1,
    coarse_time_step_s: float = timeout_audit.COARSE_TIME_STEP_S,
    fine_time_step_s: float = timeout_audit.FINE_TIME_STEP_S,
) -> None:
    timeout_audit._atomic_write_json(
        path,
        {
            "cases": [
                timeout_audit._outcome_payload(
                    outcome,
                    duration_s,
                    coarse_time_step_s,
                    fine_time_step_s,
                )
            ]
        },
    )


def _write_apply_intent(path: Path, report_path: Path) -> None:
    timeout_audit._atomic_write_json(
        path,
        {"audit_report_sha256": timeout_audit._sha256(report_path)},
    )


def test_continuation_retains_resolved_and_reruns_only_unresolved(
    tmp_path: Path,
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    resolved_case = _case(tmp_path / "resolved")
    unresolved_case = replace(
        _case(tmp_path / "unresolved"),
        point=requested_refined_points()[DETUNING_STUDY_KEY][1],
        sample=replace(_sample(), point_index=4),
    )
    resolved = _resolved_outcome(resolved_case)
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    timeout = _classification(trapped=False, reason="timeout")
    unresolved = timeout_audit.TimeoutOutcome(
        unresolved_case,
        "unresolved",
        "still timed out",
        trapped,
        trapped,
        timeout,
        timeout,
        None,
        None,
        None,
    )
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [resolved, unresolved])

    retained, rerun, retained_payloads, lineage = (
        timeout_audit._prepare_continuation(
            report_path,
            [resolved_case, unresolved_case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.2,
        )
    )

    assert retained == [resolved]
    assert rerun == [unresolved_case]
    assert list(retained_payloads) == [timeout_audit._case_key(resolved_case)]
    assert lineage["retained_resolved_case_count"] == 1
    assert lineage["rerun_case_count"] == 1
    assert lineage["prior_audit_duration_s"] == pytest.approx(0.1)


def test_continuation_reclassifies_definitive_velocity_scan_without_recompute(
    tmp_path: Path,
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = replace(
        _case(tmp_path / "case"),
        sample=_lower_timeout_sample(),
        endpoint_kind="lower",
    )
    velocities = tuple(0.25 * index for index in range(121))
    scans = tuple(
        _classification(
            trapped=velocity in {0.5, 0.75, 1.0},
            reason=(
                "bounded_core_residence"
                if velocity in {0.5, 0.75, 1.0}
                else "escaped"
            ),
        )
        for velocity in velocities
    )
    unresolved = timeout_audit.TimeoutOutcome(
        case,
        "unresolved",
        "finite-speed trapped island detected",
        scans[0],
        scans[0],
        scans[80],
        scans[80],
        None,
        None,
        None,
        velocities,
        scans,
        scans,
    )
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [unresolved])

    retained, rerun, retained_payloads, lineage = (
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.1,
        )
    )

    assert rerun == []
    assert len(retained) == 1
    assert retained[0].status == "velocity_resolved_capture"
    assert retained[0].safe_to_apply
    payload = retained_payloads[timeout_audit._case_key(case)]
    assert payload["safe_to_apply"] is True
    assert payload["reclassification_used_existing_trajectory_results"] is True
    assert lineage["velocity_resolved_reclassified_case_count"] == 1
    assert lineage["zero_recompute_velocity_resolved_reclassification"] is True
    assert lineage["rerun_case_count"] == 0


def test_continuation_rejects_shorter_duration(tmp_path: Path) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_resolved_outcome(case)])

    with pytest.raises(ValueError, match="cannot be shorter"):
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.099,
        )


def test_safe_report_can_be_reused_at_same_duration_without_recompute(
    tmp_path: Path,
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_resolved_outcome(case)])

    retained, rerun, _, lineage = timeout_audit._prepare_continuation(
        report_path,
        [case],
        paths=campaign,
        study_key=DETUNING_STUDY_KEY,
        audit_duration_s=0.1,
    )

    assert len(retained) == 1
    assert rerun == []
    assert lineage["zero_recompute_safe_report_reuse"] is True


def test_same_duration_is_refused_when_any_case_needs_rerun(tmp_path: Path) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    trapped = _classification(trapped=True, reason="bounded_core_residence")
    timeout = _classification(trapped=False, reason="timeout")
    unresolved = timeout_audit.TimeoutOutcome(
        case,
        "unresolved",
        "still timed out",
        trapped,
        trapped,
        timeout,
        timeout,
        None,
        None,
        None,
    )
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [unresolved])

    with pytest.raises(ValueError, match="strictly finer"):
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.1,
        )


def test_same_duration_both_finer_steps_reruns_only_unresolved(
    tmp_path: Path,
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_unresolved_outcome(case)])

    retained, rerun, _, lineage = timeout_audit._prepare_continuation(
        report_path,
        [case],
        paths=campaign,
        study_key=DETUNING_STUDY_KEY,
        audit_duration_s=0.1,
        coarse_time_step_s=2.5e-6,
        fine_time_step_s=1.25e-6,
    )

    assert retained == []
    assert rerun == [case]
    assert lineage["continuation_axis"] == "timestep_refinement"


@pytest.mark.parametrize(
    ("coarse_step_s", "fine_step_s"),
    [(4.0e-6, 2.5e-6), (5.0e-6, 1.25e-6)],
)
def test_same_duration_one_finer_step_is_refused(
    tmp_path: Path, coarse_step_s: float, fine_step_s: float
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_unresolved_outcome(case)])

    with pytest.raises(ValueError, match="both coarse and fine"):
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.1,
            coarse_time_step_s=coarse_step_s,
            fine_time_step_s=fine_step_s,
        )


@pytest.mark.parametrize(
    ("coarse_step_s", "fine_step_s"),
    [
        (5.0e-6, 2.5e-6),
        (4.0e-6, 2.5e-6),
        (5.0e-6, 2.0e-6),
        (2.5e-6, 1.25e-6),
    ],
)
def test_longer_duration_accepts_same_or_finer_timestep_components(
    tmp_path: Path, coarse_step_s: float, fine_step_s: float
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_unresolved_outcome(case)])

    _, rerun, _, lineage = timeout_audit._prepare_continuation(
        report_path,
        [case],
        paths=campaign,
        study_key=DETUNING_STUDY_KEY,
        audit_duration_s=0.2,
        coarse_time_step_s=coarse_step_s,
        fine_time_step_s=fine_step_s,
    )

    assert rerun == [case]
    assert lineage["continuation_axis"] in {
        "duration_extension",
        "duration_and_timestep_refinement",
    }


@pytest.mark.parametrize(
    ("coarse_step_s", "fine_step_s"),
    [(6.0e-6, 2.5e-6), (5.0e-6, 3.0e-6)],
)
def test_longer_duration_rejects_either_coarser_timestep_component(
    tmp_path: Path, coarse_step_s: float, fine_step_s: float
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_unresolved_outcome(case)])

    with pytest.raises(ValueError, match="cannot be coarser"):
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.2,
            coarse_time_step_s=coarse_step_s,
            fine_time_step_s=fine_step_s,
        )


@pytest.mark.parametrize(
    ("duration_s", "coarse_step_s", "fine_step_s"),
    [(0.11, 5.0e-6, 2.5e-6), (0.1, 4.0e-6, 2.0e-6)],
)
def test_zero_recompute_safe_reuse_rejects_any_search_setting_change(
    tmp_path: Path,
    duration_s: float,
    coarse_step_s: float,
    fine_step_s: float,
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_resolved_outcome(case)])

    with pytest.raises(ValueError, match="exact prior"):
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=duration_s,
            coarse_time_step_s=coarse_step_s,
            fine_time_step_s=fine_step_s,
        )


def test_mixed_continuation_report_preserves_prior_and_new_case_timesteps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    retained_case = _case(tmp_path / "retained")
    rerun_case = replace(
        _case(tmp_path / "rerun"),
        point=requested_refined_points()[DETUNING_STUDY_KEY][1],
        sample=replace(_sample(), point_index=4),
    )
    prior_report = tmp_path / "prior.json"
    _write_prior_report(
        prior_report,
        campaign,
        [_resolved_outcome(retained_case), _unresolved_outcome(rerun_case)],
        duration_s=0.2,
    )
    monkeypatch.setattr(
        timeout_audit, "_campaign_completion_issues", lambda *args: []
    )
    monkeypatch.setattr(
        timeout_audit,
        "scan_completed_timeouts",
        lambda *args: [retained_case, rerun_case],
    )
    monkeypatch.setattr(
        timeout_audit, "_predicted_loading_changes", lambda outcomes: []
    )

    def fake_audit_cases(cases, **kwargs):
        assert cases == [rerun_case]
        assert kwargs["audit_duration_s"] == pytest.approx(0.2)
        assert kwargs["coarse_time_step_s"] == pytest.approx(2.5e-6)
        assert kwargs["fine_time_step_s"] == pytest.approx(1.25e-6)
        return [_resolved_outcome(rerun_case)], []

    monkeypatch.setattr(timeout_audit, "_audit_cases", fake_audit_cases)

    report = timeout_audit.run_timeout_audit(
        paths=campaign,
        audit_directory=tmp_path / "continuation",
        study_key=DETUNING_STUDY_KEY,
        audit_duration_s=0.2,
        coarse_time_step_s=2.5e-6,
        fine_time_step_s=1.25e-6,
        prior_report_path=prior_report,
    )

    by_point = {payload["point_index"]: payload for payload in report["cases"]}
    assert report["safe_to_apply"] is True
    assert report["continuation_axis"] == "timestep_refinement"
    assert by_point[0]["audit_capture_search_configs"]["coarse"][
        "time_step_s"
    ] == pytest.approx(5.0e-6)
    assert by_point[0]["audit_capture_search_configs"]["fine"][
        "time_step_s"
    ] == pytest.approx(2.5e-6)
    assert by_point[1]["audit_capture_search_configs"]["coarse"][
        "time_step_s"
    ] == pytest.approx(2.5e-6)
    assert by_point[1]["audit_capture_search_configs"]["fine"][
        "time_step_s"
    ] == pytest.approx(1.25e-6)


def test_continuation_rejects_inconsistent_safe_prior_outcome(tmp_path: Path) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_resolved_outcome(case)])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["cases"][0]["coarse_upper_result"]["termination_reason"] = "timeout"
    timeout_audit._atomic_write_json(report_path, payload)

    with pytest.raises(RuntimeError, match="lacks definitive endpoints"):
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.2,
        )


def test_continuation_rejects_contradictory_trapped_boolean_and_reason(
    tmp_path: Path,
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    report_path = tmp_path / "prior.json"
    _write_prior_report(report_path, campaign, [_resolved_outcome(case)])
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["cases"][0]["coarse_lower_result"]["termination_reason"] = "escaped"
    payload["cases"][0]["coarse_lower_result"]["trapped"] = True
    timeout_audit._atomic_write_json(report_path, payload)

    with pytest.raises(RuntimeError, match="lacks definitive endpoints"):
        timeout_audit._prepare_continuation(
            report_path,
            [case],
            paths=campaign,
            study_key=DETUNING_STUDY_KEY,
            audit_duration_s=0.1,
        )


def test_audit_search_duration_is_configurable(tmp_path: Path) -> None:
    coarse, fine = timeout_audit._audit_searches(
        _case(tmp_path).run_metadata, audit_duration_s=0.2
    )

    assert coarse.max_simulation_time_s == pytest.approx(0.2)
    assert fine.max_simulation_time_s == pytest.approx(0.2)
    assert coarse.time_step_s == timeout_audit.COARSE_TIME_STEP_S
    assert fine.time_step_s == timeout_audit.FINE_TIME_STEP_S


def test_raw_outcome_payload_identifies_scan_coordinate_and_power(tmp_path: Path) -> None:
    case = replace(
        _case(tmp_path),
        point=requested_refined_points()[RAW_STUDY_KEY][0],
    )

    payload = timeout_audit._outcome_payload(_resolved_outcome(case))

    assert payload["study_key"] == RAW_STUDY_KEY
    assert payload["scan_variable"] == "s0"
    assert payload["scan_value"] == pytest.approx(0.25)
    assert payload["s0"] == pytest.approx(0.25)
    assert payload["cooling_power_w_per_beam"] == pytest.approx(
        case.point.cooling_power_w_per_beam
    )


def test_post_apply_requires_exact_replacement_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    case = _case(tmp_path / "case")
    outcome = _resolved_outcome(case)
    wrong = replace(
        outcome.replacement_sample,
        capture_velocity_m_per_s=0.15,
        trapped_velocity_lower_m_per_s=0.15,
    )
    save_samples_atomic(case.paths.partial_samples_csv, [wrong])
    save_samples_atomic(case.paths.final_samples_csv, [wrong])
    monkeypatch.setattr(
        timeout_audit, "_validate_completed_point_inputs", lambda *args: None
    )

    with pytest.raises(ValueError, match="configuration value differs"):
        timeout_audit._validate_post_apply(
            campaign, [outcome], DETUNING_STUDY_KEY
        )


def test_post_apply_matches_replacements_to_their_own_relationship_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "campaign_stats", tmp_path / "campaign_figs")
    first_case = _case(tmp_path / "first_case")
    second_case = replace(
        _case(tmp_path / "second_case"),
        point=requested_refined_points()[DETUNING_STUDY_KEY][1],
    )
    first_outcome = _resolved_outcome(first_case)
    second_outcome = _resolved_outcome(second_case)
    second_replacement = replace(
        second_outcome.replacement_sample,
        capture_velocity_m_per_s=0.4,
        trapped_velocity_lower_m_per_s=0.4,
        untrapped_velocity_upper_m_per_s=0.5,
        velocity_resolution_m_per_s=0.1,
    )
    second_outcome = replace(
        second_outcome,
        replacement_sample=second_replacement,
    )
    for case, outcome in (
        (first_case, first_outcome),
        (second_case, second_outcome),
    ):
        save_samples_atomic(case.paths.partial_samples_csv, [outcome.replacement_sample])
        save_samples_atomic(case.paths.final_samples_csv, [outcome.replacement_sample])
    aggregate = campaign.aggregate_csv(DETUNING_STUDY_KEY)
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text(
        "point_index\n"
        + "".join(
            f"{index}\n"
            for index in range(len(requested_refined_points()[DETUNING_STUDY_KEY]))
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        timeout_audit, "_validate_completed_point_inputs", lambda *args: None
    )

    timeout_audit._validate_post_apply(
        campaign,
        [first_outcome, second_outcome],
        DETUNING_STUDY_KEY,
    )


def test_campaign_apply_lock_excludes_other_studies(tmp_path: Path) -> None:
    campaign = CampaignPaths(tmp_path / "stats" / "campaign", tmp_path / "figs")
    campaign.statistics.parent.mkdir(parents=True)

    with timeout_audit._exclusive_campaign_apply_lock(campaign) as lock:
        assert lock.is_dir()
        with pytest.raises(RuntimeError, match="another timeout-audit apply"):
            with timeout_audit._exclusive_campaign_apply_lock(campaign):
                pass

    assert not lock.exists()


def test_apply_result_write_failure_rolls_back_all_production_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = CampaignPaths(tmp_path / "production_stats", tmp_path / "production_figs")
    case = _case(tmp_path / "case")
    metadata = {
        "capture_search_config": case.run_metadata["capture_search_config"],
        "run_signature_payload": {"geometry_sha256": "geometry"},
        "run_signature_sha256": "base-signature",
    }
    case.paths.metadata_json.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    save_samples_atomic(case.paths.partial_samples_csv, [case.sample])
    save_samples_atomic(case.paths.final_samples_csv, [case.sample])
    case = replace(
        case,
        run_metadata=metadata,
        source_sample_sha256=timeout_audit._sha256(case.paths.final_samples_csv),
        source_metadata_sha256=timeout_audit._sha256(case.paths.metadata_json),
    )
    outcome = _resolved_outcome(case)
    modified = [
        case.paths.partial_samples_csv,
        case.paths.final_samples_csv,
        case.paths.metadata_json,
    ]
    snapshot = timeout_audit._snapshot_files(modified, require_all=True)
    original = {path: path.read_bytes() for path in modified}
    monkeypatch.setattr(
        timeout_audit, "_campaign_completion_issues", lambda paths, study_key: []
    )
    monkeypatch.setattr(
        timeout_audit, "scan_completed_timeouts", lambda paths, study_key: [case]
    )
    monkeypatch.setattr(
        timeout_audit,
        "_modified_product_paths",
        lambda paths, outcomes, study_key: modified,
    )
    monkeypatch.setattr(
        timeout_audit,
        "analyze_completed_samples",
        lambda *args, **kwargs: {
            "loading_rate": {"loading_rate_mean_atoms_per_s": 1.0}
        },
    )
    monkeypatch.setattr(timeout_audit, "_append_provenance", lambda *args: None)
    monkeypatch.setattr(
        timeout_audit, "_rebuild_study_aggregate", lambda *args: []
    )
    monkeypatch.setattr(timeout_audit, "_validate_post_apply", lambda *args: None)
    real_atomic_write_json = timeout_audit._atomic_write_json

    def fail_success_record(path, payload):
        if path.name == timeout_audit.APPLY_RESULT_NAME:
            raise OSError("injected apply-result write failure")
        return real_atomic_write_json(path, payload)

    monkeypatch.setattr(timeout_audit, "_atomic_write_json", fail_success_record)
    audit_directory = tmp_path / "audit"
    audit_directory.mkdir()
    report_path = audit_directory / timeout_audit.AUDIT_REPORT_NAME
    intent_path = audit_directory / timeout_audit.APPLY_INTENT_NAME
    _write_apply_report(report_path, outcome)
    _write_apply_intent(intent_path, report_path)

    with pytest.raises(timeout_audit.ApplyTransactionError) as caught:
        timeout_audit._apply_corrections(
            campaign,
            [outcome],
            report_path=report_path,
            audit_directory=audit_directory,
            initial_cases=[case],
            initial_snapshot=snapshot,
            intent_path=intent_path,
            study_key=DETUNING_STUDY_KEY,
        )

    assert caught.value.rollback_complete
    assert all(path.read_bytes() == original[path] for path in modified)
