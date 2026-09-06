"""Fail-closed capture-boundary audits for the deterministic two-level MOT.

The ordinary capture search stores one scalar threshold per launch ray and is
therefore intentionally fast.  A trajectory that reaches the integration
deadline is *not* evidence that the atom escaped, however, and a zero scalar
threshold can also hide a finite-speed capture island.  This module supplies
the slower, read-only numerical checks needed before relationship-campaign
products are described as final:

* repeat saved endpoints with a longer duration at two timesteps;
* re-bisect a boundary when a formerly timed-out endpoint later traps;
* scan zero-threshold rays directly on the analysis-velocity grid; and
* retain a boolean velocity mask when capture is nonmonotone.

The module deliberately performs no filesystem mutation.  A caller must only
replace production products after every outcome is resolved and any direct
velocity masks are incorporated into the spectrum and loading integrals.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import pi
from typing import Callable, Mapping, Sequence

import numpy as np

from ..capture_statistics import CaptureVelocitySample, TrajectoryClassification
from ..configuration import AntiHelmholtzCoilConfig, MOTApparatusConfig
from ..launch_geometry import PointSample
from .batched_sampling import (
    classify_initial_conditions_batch,
    classify_trajectory_batch,
)
from .configuration import SimpleMOTConfig
from .loading import (
    LOADING_RATE_PREFACTOR,
    THERMAL_SCALE_M2_PER_S2,
    calculate_loading_rate_from_spectrum,
)
from .power_loading_study import (
    _student_t_critical_95,
    calculate_clustered_cross_section,
    calculate_disc_clustered_loading,
)
from .sampling import CaptureSearchConfig, classify_trajectory, find_capture_velocity
from .simulation import SimpleMOTBeam, build_simple_mot_beams


AUDIT_DURATION_S = 200.0e-3
COARSE_TIME_STEP_S = 5.0e-6
FINE_TIME_STEP_S = 2.5e-6
AUDIT_VELOCITY_TOLERANCE_M_PER_S = 0.249
TRAPPED_TERMINATION_REASONS = {"two_core_entries", "bounded_core_residence"}


@dataclass(frozen=True, slots=True)
class AdaptiveAuditLevel:
    """One bounded escalation used only for unresolved velocity-grid nodes."""

    duration_s: float
    coarse_time_step_s: float
    fine_time_step_s: float


DEFAULT_ADAPTIVE_AUDIT_LEVELS: tuple[AdaptiveAuditLevel, ...] = (
    AdaptiveAuditLevel(250.0e-3, 5.0e-6, 2.5e-6),
    AdaptiveAuditLevel(400.0e-3, 2.5e-6, 1.25e-6),
)

Classifier = Callable[
    [
        list[SimpleMOTBeam],
        PointSample,
        float,
        AntiHelmholtzCoilConfig,
        SimpleMOTConfig,
        CaptureSearchConfig,
    ],
    TrajectoryClassification,
]
BatchClassifier = Callable[
    [
        Sequence[SimpleMOTBeam],
        PointSample,
        Sequence[float],
        AntiHelmholtzCoilConfig,
        SimpleMOTConfig,
        CaptureSearchConfig,
    ],
    Sequence[TrajectoryClassification],
]
InitialConditionsBatchClassifier = Callable[
    [
        Sequence[SimpleMOTBeam],
        Sequence[Sequence[float]] | np.ndarray,
        Sequence[Sequence[float]] | np.ndarray,
        AntiHelmholtzCoilConfig,
        SimpleMOTConfig,
        CaptureSearchConfig,
    ],
    Sequence[TrajectoryClassification],
]
Rebisector = Callable[
    [
        list[SimpleMOTBeam],
        PointSample,
        AntiHelmholtzCoilConfig,
        SimpleMOTConfig,
        CaptureSearchConfig,
    ],
    CaptureVelocitySample,
]


@dataclass(frozen=True, slots=True)
class VelocityResolvedCaptureOverride:
    """Direct capture evidence for one exceptional, nonmonotone launch ray.

    The mask may begin captured when a 50 ms zero-speed timeout becomes a
    trapped trajectory in the authoritative 200 ms audit.  It must end
    escaped so the recorded analysis domain still brackets capture.
    """

    disc_index: int
    point_index: int
    velocity_m_per_s: tuple[float, ...]
    captured: tuple[bool, ...]

    @property
    def key(self) -> tuple[int, int]:
        return self.disc_index, self.point_index


@dataclass(frozen=True, slots=True)
class TimeoutAuditCase:
    """One saved threshold and the exact physics required to reproduce it."""

    sample: CaptureVelocitySample
    apparatus: MOTApparatusConfig
    simple_config: SimpleMOTConfig
    coil_config: AntiHelmholtzCoilConfig
    search_config: CaptureSearchConfig


@dataclass(frozen=True, slots=True)
class AdaptiveAuditEvidence:
    """Dual-timestep evidence for one node at one adaptive audit level."""

    level_index: int
    duration_s: float
    coarse_time_step_s: float
    fine_time_step_s: float
    velocity_m_per_s: float
    coarse_result: TrajectoryClassification
    fine_result: TrajectoryClassification

    @property
    def resolved(self) -> bool:
        coarse_definitive = _is_trapped(self.coarse_result) or _is_escaped(
            self.coarse_result
        )
        fine_definitive = _is_trapped(self.fine_result) or _is_escaped(
            self.fine_result
        )
        return bool(
            coarse_definitive
            and fine_definitive
            and self.coarse_result.trapped == self.fine_result.trapped
        )


@dataclass(frozen=True, slots=True)
class TimeoutAuditOutcome:
    """Read-only result of a dual-timestep boundary audit.

    Direct-grid outcomes retain the final per-node coarse and fine results.
    Nodes replaced by adaptive escalation contain their last evaluated result;
    the corresponding level configuration remains in ``adaptive_evidence``.
    Non-grid outcomes leave the three grid tuples empty.
    """

    case: TimeoutAuditCase
    status: str
    reason: str
    replacement_sample: CaptureVelocitySample | None
    velocity_override: VelocityResolvedCaptureOverride | None
    coarse_lower_result: TrajectoryClassification
    fine_lower_result: TrajectoryClassification
    coarse_upper_result: TrajectoryClassification
    fine_upper_result: TrajectoryClassification
    coarse_rebisected: CaptureVelocitySample | None = None
    fine_rebisected: CaptureVelocitySample | None = None
    adaptive_evidence: tuple[AdaptiveAuditEvidence, ...] = ()
    coarse_boundary_sample: CaptureVelocitySample | None = None
    fine_boundary_sample: CaptureVelocitySample | None = None
    velocity_grid_m_per_s: tuple[float, ...] = ()
    coarse_velocity_grid_results: tuple[TrajectoryClassification, ...] = ()
    fine_velocity_grid_results: tuple[TrajectoryClassification, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.status in {
            "confirmed_boundary",
            "confirmed_zero_capture",
            "grid_resolved_capture",
            "rebisected",
            "velocity_resolved_capture",
        } and self.replacement_sample is not None


def point_from_sample(sample: CaptureVelocitySample) -> PointSample:
    """Reconstruct the immutable launch point stored in a threshold row."""

    return PointSample(
        disc_index=sample.disc_index,
        point_index=sample.point_index,
        theta_rad=sample.theta_rad,
        phi_rad=sample.phi_rad,
        theta_prime_rad=sample.theta_prime_rad,
        s_m=sample.s_m,
        radial_distance_m=sample.radial_distance_m,
        initial_position_m=sample.initial_position_m,
        incident_unit_vector=sample.incident_unit_vector,
        launch_axis_unit_vector=sample.incident_unit_vector,
    )


def audit_searches(
    production: CaptureSearchConfig,
    *,
    duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
) -> tuple[CaptureSearchConfig, CaptureSearchConfig]:
    """Return longer coarse/fine searches with the production physics intact."""

    values = (duration_s, coarse_time_step_s, fine_time_step_s)
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("audit duration and timesteps must be finite and positive")
    if duration_s < production.max_simulation_time_s:
        raise ValueError("audit duration cannot be shorter than production")
    if coarse_time_step_s > production.time_step_s:
        raise ValueError("coarse audit timestep cannot exceed the production timestep")
    if fine_time_step_s >= coarse_time_step_s:
        raise ValueError("fine audit timestep must be strictly smaller than coarse")
    coarse = replace(
        production,
        max_simulation_time_s=float(duration_s),
        time_step_s=float(coarse_time_step_s),
        velocity_tolerance_m_per_s=AUDIT_VELOCITY_TOLERANCE_M_PER_S,
    )
    return coarse, replace(coarse, time_step_s=float(fine_time_step_s))


def _is_trapped(result: TrajectoryClassification) -> bool:
    return bool(
        result.trapped
        and result.termination_reason in TRAPPED_TERMINATION_REASONS
    )


def _is_escaped(result: TrajectoryClassification) -> bool:
    return bool(not result.trapped and result.termination_reason == "escaped")


def _classify(
    classifier: Classifier,
    beams: list[SimpleMOTBeam],
    point: PointSample,
    speed: float,
    case: TimeoutAuditCase,
    search: CaptureSearchConfig,
) -> TrajectoryClassification:
    return classifier(
        beams,
        point,
        float(speed),
        case.coil_config,
        case.simple_config,
        search,
    )


def _rebisect(
    rebisector: Rebisector,
    beams: list[SimpleMOTBeam],
    point: PointSample,
    case: TimeoutAuditCase,
    search: CaptureSearchConfig,
) -> CaptureVelocitySample:
    return rebisector(
        beams,
        point,
        case.coil_config,
        case.simple_config,
        search,
    )


def _definitive_bracket(sample: CaptureVelocitySample) -> bool:
    lower = sample.trapped_velocity_lower_m_per_s
    upper = sample.untrapped_velocity_upper_m_per_s
    width = upper - lower
    return bool(
        sample.lower_classification in TRAPPED_TERMINATION_REASONS
        and sample.upper_classification == "escaped"
        and 0.0 <= lower < upper
        and 0.0 < width < AUDIT_VELOCITY_TOLERANCE_M_PER_S + 1.0e-12
        and np.isclose(sample.capture_velocity_m_per_s, lower, atol=1.0e-12, rtol=0.0)
        and np.isclose(sample.velocity_resolution_m_per_s, width, atol=1.0e-12, rtol=0.0)
    )


def _compatible_brackets(
    coarse: CaptureVelocitySample, fine: CaptureVelocitySample
) -> bool:
    tolerance = AUDIT_VELOCITY_TOLERANCE_M_PER_S
    overlap_width = min(
        coarse.untrapped_velocity_upper_m_per_s,
        fine.untrapped_velocity_upper_m_per_s,
    ) - max(
        coarse.trapped_velocity_lower_m_per_s,
        fine.trapped_velocity_lower_m_per_s,
    )
    return bool(
        # A zero-width contact is an explicit trapped/escaped contradiction at
        # one speed, not converged boundary overlap.
        overlap_width > 1.0e-12
        and abs(
            coarse.trapped_velocity_lower_m_per_s
            - fine.trapped_velocity_lower_m_per_s
        )
        <= tolerance
        and abs(
            coarse.untrapped_velocity_upper_m_per_s
            - fine.untrapped_velocity_upper_m_per_s
        )
        <= tolerance
    )


def _scan_grid(search: CaptureSearchConfig) -> tuple[float, ...]:
    start = float(search.analysis_velocity_min_m_per_s)
    step = float(search.analysis_velocity_step_m_per_s)
    stop = float(search.analysis_velocity_max_m_per_s)
    if not all(np.isfinite(value) for value in (start, step, stop)):
        raise ValueError("analysis velocity bounds and step must be finite")
    if step <= 0.0 or stop <= 0.0:
        raise ValueError("analysis velocity step and maximum must be positive")
    if start != 0.0:
        raise ValueError("direct velocity-grid audits require a zero velocity minimum")
    if stop <= start:
        raise ValueError("analysis velocity maximum must exceed its minimum")
    return tuple(
        float(value) for value in np.arange(start, stop + 0.5 * step, step)
    )


def _replacement_from_zero_scan(
    sample: CaptureVelocitySample,
    velocities: Sequence[float],
    fine_results: Sequence[TrajectoryClassification],
) -> CaptureVelocitySample:
    first_positive = next(
        (
            index
            for index in range(1, len(velocities))
            if _is_escaped(fine_results[index])
        ),
        0,
    )
    if first_positive == 0:
        raise ValueError("zero-scan fallback has no positive escaped velocity node")
    return replace(
        sample,
        capture_velocity_m_per_s=0.0,
        velocity_resolution_m_per_s=float(velocities[first_positive]),
        trapped_velocity_lower_m_per_s=0.0,
        untrapped_velocity_upper_m_per_s=float(velocities[first_positive]),
        lower_classification=fine_results[0].termination_reason,
        upper_classification=fine_results[first_positive].termination_reason,
        lower_entered_trap_core=fine_results[0].entered_trap_core,
        upper_entered_trap_core=fine_results[first_positive].entered_trap_core,
        lower_core_entry_count=fine_results[0].core_entry_count,
        upper_core_entry_count=fine_results[first_positive].core_entry_count,
    )


def _replacement_from_contiguous_grid_capture(
    sample: CaptureVelocitySample,
    velocities: Sequence[float],
    fine_results: Sequence[TrajectoryClassification],
) -> CaptureVelocitySample:
    """Build the 0.25 m/s bracket for a grid mask captured from zero."""

    captured = tuple(_is_trapped(result) for result in fine_results)
    if not captured[0] or captured[-1]:
        raise ValueError("contiguous-grid replacement requires trapped/escaped endpoints")
    upper_index = next(index for index, value in enumerate(captured) if not value)
    lower_index = upper_index - 1
    lower = float(velocities[lower_index])
    upper = float(velocities[upper_index])
    return replace(
        sample,
        capture_velocity_m_per_s=lower,
        velocity_resolution_m_per_s=upper - lower,
        trapped_velocity_lower_m_per_s=lower,
        untrapped_velocity_upper_m_per_s=upper,
        lower_classification=fine_results[lower_index].termination_reason,
        upper_classification=fine_results[upper_index].termination_reason,
        lower_entered_trap_core=fine_results[lower_index].entered_trap_core,
        upper_entered_trap_core=fine_results[upper_index].entered_trap_core,
        lower_core_entry_count=fine_results[lower_index].core_entry_count,
        upper_core_entry_count=fine_results[upper_index].core_entry_count,
    )


def _audit_zero_grid_batched(
    case: TimeoutAuditCase,
    beams: list[SimpleMOTBeam],
    point: PointSample,
    coarse_search: CaptureSearchConfig,
    fine_search: CaptureSearchConfig,
    batch_classifier: BatchClassifier,
    scalar_classifier: Classifier,
    adaptive_levels: Sequence[AdaptiveAuditLevel],
    *,
    always_override: bool = False,
) -> TimeoutAuditOutcome:
    """Resolve a zero base threshold from one authoritative dual-dt grid.

    The same grid simultaneously supplies endpoint evidence, a monotonic
    0.25 m/s bracket when capture begins at zero, and a direct mask when the
    response contains a finite-speed island.  This avoids first integrating
    four scalar endpoints and then re-integrating the same speed domain.
    """

    sample = case.sample
    velocities = _scan_grid(coarse_search)

    def scalar_endpoint(
        speed: float, search: CaptureSearchConfig
    ) -> TrajectoryClassification:
        return _classify(
            scalar_classifier, beams, point, speed, case, search
        )

    def failure(reason: str) -> TimeoutAuditOutcome:
        # These four evaluations occur only on a broken batch contract.  They
        # populate an inspectable fail-closed outcome and never authorize data.
        return TimeoutAuditOutcome(
            case,
            "unresolved",
            reason,
            None,
            None,
            scalar_endpoint(sample.trapped_velocity_lower_m_per_s, coarse_search),
            scalar_endpoint(sample.trapped_velocity_lower_m_per_s, fine_search),
            scalar_endpoint(sample.untrapped_velocity_upper_m_per_s, coarse_search),
            scalar_endpoint(sample.untrapped_velocity_upper_m_per_s, fine_search),
        )

    def scan(
        search: CaptureSearchConfig,
        scan_velocities: Sequence[float] = velocities,
    ) -> tuple[TrajectoryClassification, ...]:
        results = tuple(
            batch_classifier(
                beams,
                point,
                scan_velocities,
                case.coil_config,
                case.simple_config,
                search,
            )
        )
        if len(results) != len(scan_velocities):
            raise ValueError(
                "batch classifier returned "
                f"{len(results)} results for {len(scan_velocities)} velocities"
            )
        if any(not isinstance(item, TrajectoryClassification) for item in results):
            raise TypeError("batch classifier must return TrajectoryClassification values")
        return results

    try:
        coarse_scan = list(scan(coarse_search))
        fine_scan = list(scan(fine_search))
    except (TypeError, ValueError, RuntimeError) as exc:
        return failure(f"velocity-grid batch classification failed closed: {exc}")

    def grid_evidence() -> dict[str, tuple[object, ...]]:
        """Snapshot the final per-node state available at this return point."""

        return {
            "velocity_grid_m_per_s": tuple(velocities),
            "coarse_velocity_grid_results": tuple(coarse_scan),
            "fine_velocity_grid_results": tuple(fine_scan),
        }

    def grid_endpoint(
        speed: float,
        results: Sequence[TrajectoryClassification],
        search: CaptureSearchConfig,
    ) -> TrajectoryClassification:
        matches = [
            index
            for index, velocity in enumerate(velocities)
            if np.isclose(velocity, speed, atol=1.0e-12, rtol=0.0)
        ]
        return results[matches[0]] if matches else scalar_endpoint(speed, search)

    def pair_resolved(
        coarse_result: TrajectoryClassification,
        fine_result: TrajectoryClassification,
    ) -> bool:
        definitive = (
            (_is_trapped(coarse_result) or _is_escaped(coarse_result))
            and (_is_trapped(fine_result) or _is_escaped(fine_result))
        )
        return bool(definitive and coarse_result.trapped == fine_result.trapped)

    unresolved_indices = [
        index
        for index, (coarse_result, fine_result) in enumerate(
            zip(coarse_scan, fine_scan, strict=True)
        )
        if not pair_resolved(coarse_result, fine_result)
    ]
    adaptive_evidence: list[AdaptiveAuditEvidence] = []
    last_coarse_search = coarse_search
    last_fine_search = fine_search
    for level_index, level in enumerate(adaptive_levels, start=1):
        if not unresolved_indices:
            break
        level_coarse, level_fine = audit_searches(
            case.search_config,
            duration_s=level.duration_s,
            coarse_time_step_s=level.coarse_time_step_s,
            fine_time_step_s=level.fine_time_step_s,
        )
        unresolved_velocities = tuple(velocities[index] for index in unresolved_indices)
        try:
            level_coarse_results = scan(level_coarse, unresolved_velocities)
            level_fine_results = scan(level_fine, unresolved_velocities)
        except (TypeError, ValueError, RuntimeError) as exc:
            lower = float(sample.trapped_velocity_lower_m_per_s)
            upper = float(sample.untrapped_velocity_upper_m_per_s)
            return TimeoutAuditOutcome(
                case,
                "unresolved",
                f"adaptive velocity-node classification failed closed: {exc}",
                None,
                None,
                grid_endpoint(lower, coarse_scan, last_coarse_search),
                grid_endpoint(lower, fine_scan, last_fine_search),
                grid_endpoint(upper, coarse_scan, last_coarse_search),
                grid_endpoint(upper, fine_scan, last_fine_search),
                adaptive_evidence=tuple(adaptive_evidence),
                **grid_evidence(),
            )
        next_unresolved: list[int] = []
        for index, coarse_result, fine_result in zip(
            unresolved_indices,
            level_coarse_results,
            level_fine_results,
            strict=True,
        ):
            evidence = AdaptiveAuditEvidence(
                level_index=level_index,
                duration_s=level.duration_s,
                coarse_time_step_s=level.coarse_time_step_s,
                fine_time_step_s=level.fine_time_step_s,
                velocity_m_per_s=velocities[index],
                coarse_result=coarse_result,
                fine_result=fine_result,
            )
            adaptive_evidence.append(evidence)
            coarse_scan[index] = coarse_result
            fine_scan[index] = fine_result
            if not evidence.resolved:
                next_unresolved.append(index)
        unresolved_indices = next_unresolved
        last_coarse_search = level_coarse
        last_fine_search = level_fine

    lower = float(sample.trapped_velocity_lower_m_per_s)
    upper = float(sample.untrapped_velocity_upper_m_per_s)
    coarse_lower = grid_endpoint(lower, coarse_scan, last_coarse_search)
    fine_lower = grid_endpoint(lower, fine_scan, last_fine_search)
    coarse_upper = grid_endpoint(upper, coarse_scan, last_coarse_search)
    fine_upper = grid_endpoint(upper, fine_scan, last_fine_search)
    if unresolved_indices:
        unresolved_speeds = ", ".join(
            f"{velocities[index]:g}" for index in unresolved_indices
        )
        return TimeoutAuditOutcome(
            case,
            "unresolved",
            "velocity-grid evidence remains non-definitive or timestep-dependent "
            f"after all adaptive levels at {unresolved_speeds} m/s",
            None,
            None,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            adaptive_evidence=tuple(adaptive_evidence),
            **grid_evidence(),
        )

    captured = tuple(result.trapped for result in fine_scan)
    if captured[-1]:
        return TimeoutAuditOutcome(
            case,
            "unresolved",
            "direct capture grid lacks an escaped high-speed endpoint",
            None,
            None,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            adaptive_evidence=tuple(adaptive_evidence),
            **grid_evidence(),
        )

    if captured[0]:
        coarse_replacement = _replacement_from_contiguous_grid_capture(
            sample, velocities, coarse_scan
        )
        replacement = _replacement_from_contiguous_grid_capture(
            sample, velocities, fine_scan
        )
        first_escape = next(index for index, value in enumerate(captured) if not value)
        has_later_island = any(captured[first_escape + 1 :])
        preserve_direct_mask = has_later_island or always_override
        override = (
            VelocityResolvedCaptureOverride(
                sample.disc_index,
                sample.point_index,
                velocities,
                captured,
            )
            if preserve_direct_mask
            else None
        )
        status = "velocity_resolved_capture" if override else "grid_resolved_capture"
        if has_later_island:
            reason = (
                "dual-timestep direct grid recovered the zero-origin capture boundary "
                "and retained later nonmonotone capture as a boolean mask"
            )
        elif always_override:
            reason = (
                "dual-timestep direct grid recovered a contiguous capture boundary "
                "and retained its boolean mask by explicit policy"
            )
        else:
            reason = (
                "dual-timestep direct grid recovered a contiguous zero-origin "
                "capture boundary"
            )
        return TimeoutAuditOutcome(
            case,
            status,
            reason,
            replacement,
            override,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            adaptive_evidence=tuple(adaptive_evidence),
            coarse_boundary_sample=coarse_replacement,
            fine_boundary_sample=replacement,
            **grid_evidence(),
        )

    coarse_replacement = _replacement_from_zero_scan(sample, velocities, coarse_scan)
    replacement = _replacement_from_zero_scan(sample, velocities, fine_scan)
    if not any(captured):
        override = (
            VelocityResolvedCaptureOverride(
                sample.disc_index,
                sample.point_index,
                velocities,
                captured,
            )
            if always_override
            else None
        )
        return TimeoutAuditOutcome(
            case,
            "velocity_resolved_capture" if override else "confirmed_zero_capture",
            (
                "dual-timestep direct velocity grid contains no capture and "
                "retained its all-false boolean mask by explicit policy"
                if override
                else "dual-timestep direct velocity scan contains no capture"
            ),
            replacement,
            override,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            adaptive_evidence=tuple(adaptive_evidence),
            coarse_boundary_sample=coarse_replacement,
            fine_boundary_sample=replacement,
            **grid_evidence(),
        )
    override = VelocityResolvedCaptureOverride(
        sample.disc_index,
        sample.point_index,
        velocities,
        captured,
    )
    return TimeoutAuditOutcome(
        case,
        "velocity_resolved_capture",
        "nonmonotone finite-speed capture retained as a direct boolean mask",
        replacement,
        override,
        coarse_lower,
        fine_lower,
        coarse_upper,
        fine_upper,
        adaptive_evidence=tuple(adaptive_evidence),
        coarse_boundary_sample=coarse_replacement,
        fine_boundary_sample=replacement,
        **grid_evidence(),
    )


def _audit_capture_boundaries_on_velocity_grid_batched(
    cases: Sequence[TimeoutAuditCase],
    *,
    require_zero_thresholds: bool,
    always_override: bool,
    initial_conditions_classifier: InitialConditionsBatchClassifier = (
        classify_initial_conditions_batch
    ),
    scalar_classifier: Classifier = classify_trajectory,
    duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
    adaptive_levels: Sequence[AdaptiveAuditLevel] = DEFAULT_ADAPTIVE_AUDIT_LEVELS,
) -> tuple[TimeoutAuditOutcome, ...]:
    """Audit several rays on a complete grid in shared vectorized integrations.

    Each physical trajectory remains an independent row.  Flattening
    ``ray x velocity`` initial conditions only amortizes the Python, NumPy, and
    elliptic-field call overhead.  The complete 0--30 m/s grid, independent
    coarse/fine RK4 integrations, adaptive node-only escalation, event order,
    and per-ray outcome construction are exactly those used by
    :func:`_audit_zero_grid_batched`.

    All cases must describe the same relationship point (apparatus, simple-MOT
    physics, coils, and search controls).  This is deliberately stricter than
    accepting merely compatible shapes so no state from different physical
    configurations can enter one integration batch.
    """

    batch_cases = tuple(cases)
    if not batch_cases:
        return ()
    first = batch_cases[0]
    keys = [
        (case.sample.disc_index, case.sample.point_index) for case in batch_cases
    ]
    if len(set(keys)) != len(keys):
        message = (
            "zero-threshold audit cases must have unique ray keys"
            if require_zero_thresholds
            else "velocity-grid audit cases must have unique ray keys"
        )
        raise ValueError(message)
    for case in batch_cases:
        if require_zero_thresholds and not np.isclose(
            case.sample.capture_velocity_m_per_s, 0.0, atol=1.0e-15, rtol=0.0
        ):
            raise ValueError("multi-ray zero-grid audit requires zero thresholds")
        if (
            case.apparatus != first.apparatus
            or case.simple_config != first.simple_config
            or case.coil_config != first.coil_config
            or case.search_config != first.search_config
        ):
            message = (
                "multi-ray zero-grid audit cases must share one physical configuration"
                if require_zero_thresholds
                else "multi-ray velocity-grid audit cases must share one physical configuration"
            )
            raise ValueError(message)

    coarse_search, fine_search = audit_searches(
        first.search_config,
        duration_s=duration_s,
        coarse_time_step_s=coarse_time_step_s,
        fine_time_step_s=fine_time_step_s,
    )
    beams = build_simple_mot_beams(first.apparatus, first.simple_config)
    points = tuple(point_from_sample(case.sample) for case in batch_cases)
    velocities = _scan_grid(coarse_search)

    def cache_key(
        case_index: int,
        search: CaptureSearchConfig,
        speeds: Sequence[float],
    ) -> tuple[int, float, float, tuple[float, ...]]:
        return (
            int(case_index),
            float(search.max_simulation_time_s),
            float(search.time_step_s),
            tuple(float(speed) for speed in speeds),
        )

    cached: dict[
        tuple[int, float, float, tuple[float, ...]],
        tuple[TrajectoryClassification, ...],
    ] = {}

    def classify_pairs(
        pairs: Sequence[tuple[int, float]],
        search: CaptureSearchConfig,
    ) -> tuple[TrajectoryClassification, ...]:
        if not pairs:
            return ()
        positions = np.asarray(
            [points[case_index].initial_position_m for case_index, _ in pairs],
            dtype=float,
        )
        directions = np.asarray(
            [points[case_index].incident_unit_vector for case_index, _ in pairs],
            dtype=float,
        )
        speeds = np.asarray([speed for _, speed in pairs], dtype=float)
        results = tuple(
            initial_conditions_classifier(
                beams,
                positions,
                speeds[:, None] * directions,
                first.coil_config,
                first.simple_config,
                search,
            )
        )
        if len(results) != len(pairs):
            raise ValueError(
                "initial-condition batch classifier returned "
                f"{len(results)} results for {len(pairs)} states"
            )
        if any(not isinstance(item, TrajectoryClassification) for item in results):
            raise TypeError(
                "initial-condition batch classifier must return "
                "TrajectoryClassification values"
            )
        return results

    full_pairs = tuple(
        (case_index, speed)
        for case_index in range(len(batch_cases))
        for speed in velocities
    )
    coarse_flat = classify_pairs(full_pairs, coarse_search)
    fine_flat = classify_pairs(full_pairs, fine_search)
    velocity_count = len(velocities)
    coarse_working: list[list[TrajectoryClassification]] = []
    fine_working: list[list[TrajectoryClassification]] = []
    for case_index in range(len(batch_cases)):
        start = case_index * velocity_count
        stop = start + velocity_count
        coarse_scan = tuple(coarse_flat[start:stop])
        fine_scan = tuple(fine_flat[start:stop])
        cached[cache_key(case_index, coarse_search, velocities)] = coarse_scan
        cached[cache_key(case_index, fine_search, velocities)] = fine_scan
        coarse_working.append(list(coarse_scan))
        fine_working.append(list(fine_scan))

    def pair_resolved(
        coarse_result: TrajectoryClassification,
        fine_result: TrajectoryClassification,
    ) -> bool:
        definitive = (
            (_is_trapped(coarse_result) or _is_escaped(coarse_result))
            and (_is_trapped(fine_result) or _is_escaped(fine_result))
        )
        return bool(definitive and coarse_result.trapped == fine_result.trapped)

    unresolved: dict[int, list[int]] = {
        case_index: [
            velocity_index
            for velocity_index, (coarse_result, fine_result) in enumerate(
                zip(
                    coarse_working[case_index],
                    fine_working[case_index],
                    strict=True,
                )
            )
            if not pair_resolved(coarse_result, fine_result)
        ]
        for case_index in range(len(batch_cases))
    }

    # Precompute every adaptive level across all rays that still need that
    # level.  The cached per-ray slices are then replayed through the original
    # single-ray policy routine, so its evidence and replacement logic remain
    # the sole authority.
    for level in adaptive_levels:
        active = {
            case_index: indices
            for case_index, indices in unresolved.items()
            if indices
        }
        if not active:
            break
        level_coarse, level_fine = audit_searches(
            first.search_config,
            duration_s=level.duration_s,
            coarse_time_step_s=level.coarse_time_step_s,
            fine_time_step_s=level.fine_time_step_s,
        )
        level_pairs = tuple(
            (case_index, velocities[velocity_index])
            for case_index, indices in active.items()
            for velocity_index in indices
        )
        level_coarse_flat = classify_pairs(level_pairs, level_coarse)
        level_fine_flat = classify_pairs(level_pairs, level_fine)
        cursor = 0
        for case_index, indices in active.items():
            count = len(indices)
            speeds = tuple(velocities[index] for index in indices)
            coarse_slice = tuple(level_coarse_flat[cursor : cursor + count])
            fine_slice = tuple(level_fine_flat[cursor : cursor + count])
            cursor += count
            cached[cache_key(case_index, level_coarse, speeds)] = coarse_slice
            cached[cache_key(case_index, level_fine, speeds)] = fine_slice
            next_unresolved: list[int] = []
            for velocity_index, coarse_result, fine_result in zip(
                indices, coarse_slice, fine_slice, strict=True
            ):
                coarse_working[case_index][velocity_index] = coarse_result
                fine_working[case_index][velocity_index] = fine_result
                if not pair_resolved(coarse_result, fine_result):
                    next_unresolved.append(velocity_index)
            unresolved[case_index] = next_unresolved

    outcomes: list[TimeoutAuditOutcome] = []
    for case_index, (case, point) in enumerate(zip(batch_cases, points, strict=True)):

        def cached_batch_classifier(
            _beams,
            _point,
            speeds,
            _coil,
            _simple,
            search,
            *,
            _case_index: int = case_index,
        ):
            key = cache_key(_case_index, search, speeds)
            if key not in cached:
                raise RuntimeError(
                    "multi-ray audit replay requested an uncomputed integration batch"
                )
            return cached[key]

        outcomes.append(
            _audit_zero_grid_batched(
                case,
                beams,
                point,
                coarse_search,
                fine_search,
                cached_batch_classifier,
                scalar_classifier,
                adaptive_levels,
                always_override=always_override,
            )
        )
    return tuple(outcomes)


def audit_zero_capture_boundaries_batched(
    cases: Sequence[TimeoutAuditCase],
    *,
    initial_conditions_classifier: InitialConditionsBatchClassifier = (
        classify_initial_conditions_batch
    ),
    scalar_classifier: Classifier = classify_trajectory,
    duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
    adaptive_levels: Sequence[AdaptiveAuditLevel] = DEFAULT_ADAPTIVE_AUDIT_LEVELS,
) -> tuple[TimeoutAuditOutcome, ...]:
    """Audit several zero-threshold rays in shared vectorized integrations."""

    return _audit_capture_boundaries_on_velocity_grid_batched(
        cases,
        require_zero_thresholds=True,
        always_override=False,
        initial_conditions_classifier=initial_conditions_classifier,
        scalar_classifier=scalar_classifier,
        duration_s=duration_s,
        coarse_time_step_s=coarse_time_step_s,
        fine_time_step_s=fine_time_step_s,
        adaptive_levels=adaptive_levels,
    )


def audit_capture_boundaries_on_velocity_grid_batched(
    cases: Sequence[TimeoutAuditCase],
    *,
    always_override: bool = False,
    initial_conditions_classifier: InitialConditionsBatchClassifier = (
        classify_initial_conditions_batch
    ),
    scalar_classifier: Classifier = classify_trajectory,
    duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
    adaptive_levels: Sequence[AdaptiveAuditLevel] = DEFAULT_ADAPTIVE_AUDIT_LEVELS,
) -> tuple[TimeoutAuditOutcome, ...]:
    """Audit arbitrary saved thresholds on the complete analysis-velocity grid.

    Every grid speed is classified independently at the requested coarse and
    fine audit settings. Only nodes that remain non-definitive or disagree are
    advanced through the bounded adaptive hierarchy. An unresolved node or a
    captured high-speed endpoint leaves the outcome unresolved. Set
    ``always_override`` to retain an agreed direct boolean mask even when the
    mask is a monotone captured prefix; this is appropriate when a prior scalar
    boundary search has already demonstrated timestep sensitivity.
    """

    return _audit_capture_boundaries_on_velocity_grid_batched(
        cases,
        require_zero_thresholds=False,
        always_override=always_override,
        initial_conditions_classifier=initial_conditions_classifier,
        scalar_classifier=scalar_classifier,
        duration_s=duration_s,
        coarse_time_step_s=coarse_time_step_s,
        fine_time_step_s=fine_time_step_s,
        adaptive_levels=adaptive_levels,
    )


def audit_capture_boundary(
    case: TimeoutAuditCase,
    *,
    classifier: Classifier = classify_trajectory,
    batch_classifier: BatchClassifier | None = None,
    rebisector: Rebisector = find_capture_velocity,
    duration_s: float = AUDIT_DURATION_S,
    coarse_time_step_s: float = COARSE_TIME_STEP_S,
    fine_time_step_s: float = FINE_TIME_STEP_S,
    scan_zero_threshold: bool = True,
    adaptive_levels: Sequence[AdaptiveAuditLevel] = DEFAULT_ADAPTIVE_AUDIT_LEVELS,
) -> TimeoutAuditOutcome:
    """Audit one threshold without modifying any production artifact.

    Set ``scan_zero_threshold`` for every saved zero threshold.  This is
    required even when its endpoints are timeout-free because a scalar binary
    search cannot rule out a finite-speed capture island.
    """

    sample = case.sample
    has_timeout = "timeout" in (
        sample.lower_classification,
        sample.upper_classification,
    )
    has_nonfinite = "non_finite" in (
        sample.lower_classification,
        sample.upper_classification,
    )
    is_zero = np.isclose(sample.capture_velocity_m_per_s, 0.0, atol=1.0e-15)
    if not (has_timeout or has_nonfinite or (scan_zero_threshold and is_zero)):
        raise ValueError("case has neither an unresolved endpoint nor a zero threshold")

    coarse_search, fine_search = audit_searches(
        case.search_config,
        duration_s=duration_s,
        coarse_time_step_s=coarse_time_step_s,
        fine_time_step_s=fine_time_step_s,
    )
    beams = build_simple_mot_beams(case.apparatus, case.simple_config)
    point = point_from_sample(sample)
    lower = float(sample.trapped_velocity_lower_m_per_s)
    upper = float(sample.untrapped_velocity_upper_m_per_s)
    selected_batch_classifier = batch_classifier
    if selected_batch_classifier is None and classifier is classify_trajectory:
        selected_batch_classifier = classify_trajectory_batch
    if is_zero and scan_zero_threshold and selected_batch_classifier is not None:
        return _audit_zero_grid_batched(
            case,
            beams,
            point,
            coarse_search,
            fine_search,
            selected_batch_classifier,
            classifier,
            adaptive_levels,
        )
    coarse_lower = _classify(classifier, beams, point, lower, case, coarse_search)
    fine_lower = _classify(classifier, beams, point, lower, case, fine_search)
    coarse_upper = _classify(classifier, beams, point, upper, case, coarse_search)
    fine_upper = _classify(classifier, beams, point, upper, case, fine_search)

    if any(
        result.termination_reason == "non_finite"
        for result in (coarse_lower, fine_lower, coarse_upper, fine_upper)
    ):
        return TimeoutAuditOutcome(
            case,
            "unresolved",
            "an endpoint produced a non-finite trajectory",
            None,
            None,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
        )

    if is_zero and scan_zero_threshold:
        if _is_trapped(coarse_lower) and _is_trapped(fine_lower):
            coarse_rebisected = _rebisect(
                rebisector, beams, point, case, coarse_search
            )
            fine_rebisected = _rebisect(rebisector, beams, point, case, fine_search)
            if _definitive_bracket(coarse_rebisected) and _definitive_bracket(
                fine_rebisected
            ) and _compatible_brackets(coarse_rebisected, fine_rebisected):
                return TimeoutAuditOutcome(
                    case,
                    "rebisected",
                    "zero endpoint later traps; compatible dual-timestep brackets recovered",
                    fine_rebisected,
                    None,
                    coarse_lower,
                    fine_lower,
                    coarse_upper,
                    fine_upper,
                    coarse_rebisected,
                    fine_rebisected,
                    coarse_boundary_sample=coarse_rebisected,
                    fine_boundary_sample=fine_rebisected,
                )
            return TimeoutAuditOutcome(
                case,
                "unresolved",
                "zero endpoint traps but dual-timestep re-bisections disagree",
                None,
                None,
                coarse_lower,
                fine_lower,
                coarse_upper,
                fine_upper,
                coarse_rebisected,
                fine_rebisected,
            )
        if not (_is_escaped(coarse_lower) and _is_escaped(fine_lower)):
            return TimeoutAuditOutcome(
                case,
                "unresolved",
                "zero endpoint is not definitively escaped at both timesteps",
                None,
                None,
                coarse_lower,
                fine_lower,
                coarse_upper,
                fine_upper,
            )

        velocities = _scan_grid(coarse_search)

        # Production scans advance all speeds for this launch ray in one
        # vectorized RK4 loop.  This is algebraically the same force law and
        # terminal-event ordering as the scalar classifier, but avoids 121
        # separate Python integration loops at each audit timestep.  Tests and
        # diagnostic callers that inject a scalar classifier retain scalar
        # behavior unless they explicitly inject a matching batch callable.
        def scan(search: CaptureSearchConfig) -> tuple[TrajectoryClassification, ...]:
            if selected_batch_classifier is None:
                results = tuple(
                    _classify(classifier, beams, point, speed, case, search)
                    for speed in velocities
                )
            else:
                results = tuple(
                    selected_batch_classifier(
                        beams,
                        point,
                        velocities,
                        case.coil_config,
                        case.simple_config,
                        search,
                    )
                )
            if len(results) != len(velocities):
                raise ValueError(
                    "batch classifier returned "
                    f"{len(results)} results for {len(velocities)} velocities"
                )
            if any(not isinstance(item, TrajectoryClassification) for item in results):
                raise TypeError(
                    "batch classifier must return TrajectoryClassification values"
                )
            return results

        try:
            coarse_scan = scan(coarse_search)
            fine_scan = scan(fine_search)
        except (TypeError, ValueError, RuntimeError) as exc:
            return TimeoutAuditOutcome(
                case,
                "unresolved",
                f"velocity-grid batch classification failed closed: {exc}",
                None,
                None,
                coarse_lower,
                fine_lower,
                coarse_upper,
                fine_upper,
            )
        grid_evidence = {
            "velocity_grid_m_per_s": tuple(velocities),
            "coarse_velocity_grid_results": tuple(coarse_scan),
            "fine_velocity_grid_results": tuple(fine_scan),
        }
        for speed, coarse_result, fine_result in zip(
            velocities, coarse_scan, fine_scan, strict=True
        ):
            definitive = (
                (_is_trapped(coarse_result) or _is_escaped(coarse_result))
                and (_is_trapped(fine_result) or _is_escaped(fine_result))
            )
            if not definitive or coarse_result.trapped != fine_result.trapped:
                return TimeoutAuditOutcome(
                    case,
                    "unresolved",
                    f"velocity-grid evidence is non-definitive or timestep-dependent at {speed:g} m/s",
                    None,
                    None,
                    coarse_lower,
                    fine_lower,
                    coarse_upper,
                    fine_upper,
                    **grid_evidence,
                )
        coarse_replacement = _replacement_from_zero_scan(
            sample, velocities, coarse_scan
        )
        replacement = _replacement_from_zero_scan(sample, velocities, fine_scan)
        captured = tuple(result.trapped for result in fine_scan)
        if not any(captured):
            return TimeoutAuditOutcome(
                case,
                "confirmed_zero_capture",
                "dual-timestep direct velocity scan contains no capture",
                replacement,
                None,
                coarse_lower,
                fine_lower,
                coarse_upper,
                fine_upper,
                coarse_boundary_sample=coarse_replacement,
                fine_boundary_sample=replacement,
                **grid_evidence,
            )
        if captured[0] or captured[-1]:
            return TimeoutAuditOutcome(
                case,
                "unresolved",
                "direct capture mask lacks escaped low/high-speed endpoints",
                None,
                None,
                coarse_lower,
                fine_lower,
                coarse_upper,
                fine_upper,
                **grid_evidence,
            )
        override = VelocityResolvedCaptureOverride(
            sample.disc_index,
            sample.point_index,
            velocities,
            captured,
        )
        return TimeoutAuditOutcome(
            case,
            "velocity_resolved_capture",
            "nonmonotone finite-speed capture retained as a direct boolean mask",
            replacement,
            override,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            coarse_boundary_sample=coarse_replacement,
            fine_boundary_sample=replacement,
            **grid_evidence,
        )

    endpoints_are_definitive = bool(
        _is_trapped(coarse_lower)
        and _is_trapped(fine_lower)
        and _is_escaped(coarse_upper)
        and _is_escaped(fine_upper)
    )
    if endpoints_are_definitive:
        coarse_replacement = replace(
            sample,
            lower_classification=coarse_lower.termination_reason,
            upper_classification="escaped",
            lower_entered_trap_core=coarse_lower.entered_trap_core,
            upper_entered_trap_core=coarse_upper.entered_trap_core,
            lower_core_entry_count=coarse_lower.core_entry_count,
            upper_core_entry_count=coarse_upper.core_entry_count,
        )
        replacement = replace(
            sample,
            lower_classification=fine_lower.termination_reason,
            upper_classification="escaped",
            lower_entered_trap_core=fine_lower.entered_trap_core,
            upper_entered_trap_core=fine_upper.entered_trap_core,
            lower_core_entry_count=fine_lower.core_entry_count,
            upper_core_entry_count=fine_upper.core_entry_count,
        )
        return TimeoutAuditOutcome(
            case,
            "confirmed_boundary",
            "saved bracket is trapped/escaped at both audit timesteps",
            replacement,
            None,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            coarse_boundary_sample=coarse_replacement,
            fine_boundary_sample=replacement,
        )

    coarse_rebisected = _rebisect(rebisector, beams, point, case, coarse_search)
    fine_rebisected = _rebisect(rebisector, beams, point, case, fine_search)
    if _definitive_bracket(coarse_rebisected) and _definitive_bracket(
        fine_rebisected
    ) and _compatible_brackets(coarse_rebisected, fine_rebisected):
        return TimeoutAuditOutcome(
            case,
            "rebisected",
            "fresh dual-timestep searches recovered compatible trapped/escaped brackets",
            fine_rebisected,
            None,
            coarse_lower,
            fine_lower,
            coarse_upper,
            fine_upper,
            coarse_rebisected,
            fine_rebisected,
            coarse_boundary_sample=coarse_rebisected,
            fine_boundary_sample=fine_rebisected,
        )
    return TimeoutAuditOutcome(
        case,
        "unresolved",
        "fresh dual-timestep searches did not recover compatible definitive brackets",
        None,
        None,
        coarse_lower,
        fine_lower,
        coarse_upper,
        fine_upper,
        coarse_rebisected,
        fine_rebisected,
    )


def validate_velocity_overrides(
    samples: Sequence[CaptureVelocitySample],
    velocity_grid_m_per_s: Sequence[float],
    overrides: Sequence[VelocityResolvedCaptureOverride],
) -> dict[tuple[int, int], VelocityResolvedCaptureOverride]:
    """Validate direct masks before they are used in any aggregate statistic."""

    expected = np.asarray(velocity_grid_m_per_s, dtype=float)
    if expected.ndim != 1 or len(expected) < 2 or np.any(np.diff(expected) <= 0.0):
        raise ValueError("velocity grid must be strictly increasing")
    sample_keys = {(sample.disc_index, sample.point_index) for sample in samples}
    keyed: dict[tuple[int, int], VelocityResolvedCaptureOverride] = {}
    for override in overrides:
        velocity = np.asarray(override.velocity_m_per_s, dtype=float)
        if override.key in keyed:
            raise ValueError(f"duplicate velocity override {override.key}")
        if override.key not in sample_keys:
            raise ValueError(f"velocity override has no sample {override.key}")
        if len(override.captured) != len(expected) or not np.allclose(
            velocity, expected, atol=1.0e-12, rtol=0.0
        ):
            raise ValueError(f"velocity override {override.key} does not cover the exact grid")
        if override.captured[-1]:
            raise ValueError(
                f"velocity override {override.key} lacks an escaped high endpoint"
            )
        keyed[override.key] = override
    return keyed


def capture_predicate_with_overrides(
    samples: Sequence[CaptureVelocitySample],
    velocity_grid_m_per_s: Sequence[float],
    overrides: Sequence[VelocityResolvedCaptureOverride],
) -> Callable[[CaptureVelocitySample, float], bool]:
    """Return the threshold/direct-mask capture predicate used by reanalysis."""

    keyed = validate_velocity_overrides(samples, velocity_grid_m_per_s, overrides)
    grid = np.asarray(velocity_grid_m_per_s, dtype=float)

    def captured(sample: CaptureVelocitySample, speed: float) -> bool:
        override = keyed.get((sample.disc_index, sample.point_index))
        if override is not None:
            matches = np.flatnonzero(np.isclose(grid, speed, atol=1.0e-12, rtol=0.0))
            if len(matches) != 1:
                raise ValueError(f"speed {speed:g} m/s is absent from the direct mask")
            return bool(override.captured[int(matches[0])])
        return bool(
            sample.lower_classification in TRAPPED_TERMINATION_REASONS
            and sample.capture_velocity_m_per_s >= speed - 1.0e-12
        )

    return captured


def velocity_overrides_to_payload(
    overrides: Sequence[VelocityResolvedCaptureOverride],
) -> dict[str, object]:
    """Serialize exact direct masks in a stable, JSON-ready representation."""

    ordered = sorted(overrides, key=lambda item: item.key)
    if len({item.key for item in ordered}) != len(ordered):
        raise ValueError("duplicate velocity-resolved override")
    return {
        "schema_version": 1,
        "representation": "direct_boolean_capture_mask",
        "override_count": len(ordered),
        "overrides": [
            {
                "disc_index": item.disc_index,
                "point_index": item.point_index,
                "velocity_grid_m_per_s": list(item.velocity_m_per_s),
                "captured_mask": list(item.captured),
            }
            for item in ordered
        ],
    }


def velocity_overrides_from_payload(
    payload: Mapping[str, object],
) -> list[VelocityResolvedCaptureOverride]:
    """Deserialize masks, rejecting malformed or duplicate records."""

    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("unsupported velocity-override schema")
    if payload.get("representation") != "direct_boolean_capture_mask":
        raise ValueError("unsupported velocity-override representation")
    records = payload.get("overrides")
    if not isinstance(records, list):
        raise ValueError("velocity-override payload has no override list")
    overrides: list[VelocityResolvedCaptureOverride] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("velocity-override record is not a mapping")
        raw_velocity = record.get("velocity_grid_m_per_s")
        raw_captured = record.get("captured_mask")
        if not isinstance(raw_velocity, list) or not isinstance(raw_captured, list):
            raise ValueError("velocity-override arrays are missing")
        if any(type(value) is not bool for value in raw_captured):
            raise ValueError("captured_mask must contain JSON booleans")
        override = VelocityResolvedCaptureOverride(
            disc_index=int(record["disc_index"]),
            point_index=int(record["point_index"]),
            velocity_m_per_s=tuple(float(value) for value in raw_velocity),
            captured=tuple(raw_captured),
        )
        overrides.append(override)
    if int(payload.get("override_count", -1)) != len(overrides):
        raise ValueError("velocity-override count does not match its records")
    if len({item.key for item in overrides}) != len(overrides):
        raise ValueError("duplicate velocity-resolved override")
    return sorted(overrides, key=lambda item: item.key)


def adaptive_audit_evidence_to_payload(
    evidence: Sequence[AdaptiveAuditEvidence],
) -> list[dict[str, object]]:
    """Return complete JSON-ready provenance for adaptive node integrations."""

    def result_payload(result: TrajectoryClassification) -> dict[str, object]:
        return {
            "trapped": result.trapped,
            "termination_reason": result.termination_reason,
            "elapsed_time_s": result.elapsed_time_s,
            "minimum_radius_m": result.minimum_radius_m,
            "final_radius_m": result.final_radius_m,
            "final_position_m": list(result.final_position_m),
            "final_velocity_m_per_s": list(result.final_velocity_m_per_s),
            "entered_trap_core": result.entered_trap_core,
            "core_entry_count": result.core_entry_count,
        }

    return [
        {
            "level_index": item.level_index,
            "duration_s": item.duration_s,
            "coarse_time_step_s": item.coarse_time_step_s,
            "fine_time_step_s": item.fine_time_step_s,
            "velocity_m_per_s": item.velocity_m_per_s,
            "resolved": item.resolved,
            "coarse_result": result_payload(item.coarse_result),
            "fine_result": result_payload(item.fine_result),
        }
        for item in evidence
    ]


def calculate_clustered_cross_section_with_overrides(
    samples: Sequence[CaptureVelocitySample],
    search: CaptureSearchConfig,
    overrides: Sequence[VelocityResolvedCaptureOverride],
    velocity_grid_m_per_s: Sequence[float],
) -> list[dict[str, int | float]]:
    """Calculate disc-clustered cross sections using direct masks when present."""

    velocity = np.asarray(velocity_grid_m_per_s, dtype=float)
    if not overrides:
        return calculate_clustered_cross_section(samples, search, velocity)
    if not samples:
        raise ValueError("at least one capture sample is required")
    captured_at = capture_predicate_with_overrides(samples, velocity, overrides)
    grouped: dict[int, list[CaptureVelocitySample]] = {}
    for sample in samples:
        grouped.setdefault(sample.disc_index, []).append(sample)
    disc_ids = sorted(grouped)
    disc_count = len(disc_ids)
    if not disc_count:
        raise ValueError("at least one direction-disc cluster is required")
    area = pi * search.disc_radius_m**2
    t_critical = _student_t_critical_95(disc_count)
    rows: list[dict[str, int | float]] = []
    for speed in velocity:
        disc_cross_sections = area * np.asarray(
            [
                np.mean(
                    [captured_at(sample, float(speed)) for sample in grouped[disc_index]]
                )
                for disc_index in disc_ids
            ],
            dtype=float,
        )
        mean = float(np.mean(disc_cross_sections))
        sample_std = (
            float(np.std(disc_cross_sections, ddof=1)) if disc_count > 1 else 0.0
        )
        sem = sample_std / np.sqrt(disc_count)
        half_width = t_critical * sem
        captured_count = int(
            sum(captured_at(sample, float(speed)) for sample in samples)
        )
        rows.append(
            {
                "velocity_m_per_s": float(speed),
                "captured_count": captured_count,
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
    search: CaptureSearchConfig,
    spectrum_rows: Sequence[Mapping[str, int | float]],
    overrides: Sequence[VelocityResolvedCaptureOverride],
) -> tuple[list[dict[str, int | float]], dict[str, int | float | str]]:
    """Integrate one spectrum per disc while honoring direct capture masks."""

    if not overrides:
        return calculate_disc_clustered_loading(samples, search, spectrum_rows)
    if not spectrum_rows:
        raise ValueError("capture-cross-section spectrum must not be empty")
    velocity = np.asarray(
        [float(row["velocity_m_per_s"]) for row in spectrum_rows], dtype=float
    )
    captured_at = capture_predicate_with_overrides(samples, velocity, overrides)
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
        [float(row["loading_rate_atoms_per_s"]) for row in by_disc], dtype=float
    )
    integrals = np.asarray(
        [float(row["loading_integral_m6_per_s4"]) for row in by_disc], dtype=float
    )
    mean_sigma = np.asarray(
        [float(row["capture_cross_section_m2"]) for row in spectrum_rows],
        dtype=float,
    )
    mean_spectrum_result = calculate_loading_rate_from_spectrum(velocity, mean_sigma)
    disc_count = len(by_disc)
    sample_std = float(np.std(rates, ddof=1)) if disc_count > 1 else 0.0
    sem = sample_std / np.sqrt(disc_count)
    t_critical = _student_t_critical_95(disc_count)
    mean_rate = float(np.mean(rates))
    half_width = t_critical * sem
    return by_disc, {
        "loading_rate_mean_atoms_per_s": mean_rate,
        "loading_rate_from_mean_spectrum_atoms_per_s": (
            mean_spectrum_result.loading_rate_atoms_per_s
        ),
        "loading_rate_sample_std_atoms_per_s": sample_std,
        "loading_rate_disc_cluster_sem_atoms_per_s": sem,
        "loading_rate_t95_lower_atoms_per_s": max(0.0, mean_rate - half_width),
        "loading_rate_t95_upper_atoms_per_s": mean_rate + half_width,
        "student_t_critical_95": t_critical,
        "confidence_level": 0.95,
        "disc_count": disc_count,
        "point_count": len(samples),
        "loading_integral_mean_m6_per_s4": float(np.mean(integrals)),
        "loading_integral_from_mean_spectrum_m6_per_s4": (
            mean_spectrum_result.integral_value_m5_per_s4
        ),
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
    "AdaptiveAuditEvidence",
    "AdaptiveAuditLevel",
    "AUDIT_DURATION_S",
    "AUDIT_VELOCITY_TOLERANCE_M_PER_S",
    "COARSE_TIME_STEP_S",
    "DEFAULT_ADAPTIVE_AUDIT_LEVELS",
    "FINE_TIME_STEP_S",
    "TimeoutAuditCase",
    "TimeoutAuditOutcome",
    "VelocityResolvedCaptureOverride",
    "adaptive_audit_evidence_to_payload",
    "audit_capture_boundary",
    "audit_capture_boundaries_on_velocity_grid_batched",
    "audit_searches",
    "audit_zero_capture_boundaries_batched",
    "calculate_clustered_cross_section_with_overrides",
    "calculate_disc_clustered_loading_with_overrides",
    "capture_predicate_with_overrides",
    "point_from_sample",
    "validate_velocity_overrides",
    "velocity_overrides_from_payload",
    "velocity_overrides_to_payload",
]
