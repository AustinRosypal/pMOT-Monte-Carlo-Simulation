"""Read-only final QA for the refined two-level MOT comparison campaign.

This module does not run trajectories, regenerate figures, or modify campaign
products.  It independently reconstructs the loading statistics from the
saved per-ray thresholds and velocity-resolved masks, while also invoking the
production fail-closed audit-ledger validator as a second line of defense.

The command exits nonzero at the first failed invariant and prints a compact
JSON PASS report on success::

    python -m pmot.mot_simple.final_campaign_qa \
        --statistics-dir /path/to/campaign/statistics \
        --figures-dir /path/to/campaign/figures

The expected campaign is deliberately narrow: 67 independent loading points
(24 raw-saturation, 20 effective-saturation, and 23 detuning points), each
with 25 full-sphere direction discs and 25 uniform-area points per disc.  The
force products contain 111 deterministic detuning points.  Temperature output
is explicitly forbidden for this effective two-level mean-force comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import pi
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from . import refined_relationship_campaign as campaign
from . import simple_force_sweep as force_sweep
from .power_loading_study import (
    SAMPLE_FIELDNAMES,
    geometry_csv_text,
    geometry_rows,
    validate_checkpoint_samples,
)
from .sampling import CaptureSearchConfig, CaptureVelocitySample, load_capture_velocity_samples
from .timeout_audit import velocity_overrides_from_payload


EXPECTED_RAW_SATURATION: tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    45.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    110.0,
    120.0,
    125.0,
)
EXPECTED_EFFECTIVE_SATURATION: tuple[float, ...] = tuple(
    0.25 * index for index in range(1, 21)
)
EXPECTED_LOADING_DETUNING_N: tuple[float, ...] = tuple(
    -0.25 * index for index in range(2, 25)
)
EXPECTED_FORCE_DETUNING_N: tuple[float, ...] = tuple(
    round(-0.5 - 0.05 * index, 12) for index in range(111)
)

EXPECTED_GEOMETRY_SHA256 = (
    "02509217f582bc1619712cd31de3fcb34aac11b208c54a4cb228694f36603e17"
)
EXPECTED_DISC_COUNT = 25
EXPECTED_POINTS_PER_DISC = 25
EXPECTED_RAY_COUNT_PER_POINT = EXPECTED_DISC_COUNT * EXPECTED_POINTS_PER_DISC
EXPECTED_LOADING_POINT_COUNT = 67
EXPECTED_TOTAL_RAY_COUNT = EXPECTED_LOADING_POINT_COUNT * EXPECTED_RAY_COUNT_PER_POINT
EXPECTED_DISC_RADIUS_M = 15.0e-3
EXPECTED_SEED = 20260903
EXPECTED_REFERENCE_POWER_W = 27.0e-3
EXPECTED_BEAM_DIAMETER_M = 12.7e-3
EXPECTED_AXIAL_GRADIENT_G_PER_CM = 10.0
EXPECTED_T_CRITICAL_95_DF24 = 2.0638985616280245
EXPECTED_CONFIDENCE = 0.95
EXPECTED_VELOCITY_GRID = np.arange(0.0, 30.0 + 0.125, 0.25, dtype=float)
TRAPPED_REASONS = frozenset({"two_core_entries", "bounded_core_residence"})
UNRESOLVED_REASONS = frozenset({"timeout", "non_finite"})
LOADING_RATE_PREFACTOR = 9.1196e5
THERMAL_SCALE_M2_PER_S2 = 5.667e4

STUDY_SPECS: tuple[tuple[str, str, tuple[float, ...]], ...] = (
    (campaign.RAW_STUDY_KEY, "s0", EXPECTED_RAW_SATURATION),
    (
        campaign.EFFECTIVE_STUDY_KEY,
        "seff",
        EXPECTED_EFFECTIVE_SATURATION,
    ),
    (
        campaign.DETUNING_STUDY_KEY,
        "detuning_n",
        EXPECTED_LOADING_DETUNING_N,
    ),
)

FINAL_FIGURE_RELATIVE_PATHS: tuple[Path, ...] = (
    Path(campaign.RAW_STUDY_KEY) / "loading_rate_vs_saturation_parameter.png",
    Path(campaign.EFFECTIVE_STUDY_KEY)
    / "loading_rate_vs_effective_saturation_parameter.png",
    Path(campaign.DETUNING_STUDY_KEY) / "loading_rate_vs_detuning.png",
    Path("04_force_vs_detuning_27mW") / "damping_turnaround_vs_detuning.png",
    Path("04_force_vs_detuning_27mW") / "restoring_slope_vs_detuning.png",
)


class CampaignQAFailure(RuntimeError):
    """A fail-closed campaign-product validation error."""


@dataclass(frozen=True, slots=True)
class PointQAResult:
    study_key: str
    point_index: int
    ray_count: int
    audit_row_count: int
    override_count: int
    indeterminate_zero_flux_ray_count: int
    maximum_numeric_residual: float
    loading_rate_atoms_per_s: float


@dataclass(frozen=True, slots=True)
class CampaignQAResult:
    outcome: str
    campaign_name: str
    campaign_signature_sha256: str
    geometry_sha256: str
    loading_point_count: int
    capture_sample_row_count: int
    endpoint_audit_row_count: int
    velocity_override_count: int
    indeterminate_zero_flux_ray_count: int
    maximum_numeric_residual: float
    maximum_loading_estimator_difference_atoms_per_s: float
    force_point_count: int
    force_axis_check_count: int
    temperature_output_count: int
    final_figure_count: int
    final_figures: tuple[str, ...]


def _fail(message: str) -> None:
    raise CampaignQAFailure(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _read_json(path: Path) -> object:
    _require(path.is_file(), f"required JSON is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignQAFailure(f"cannot read valid JSON from {path}: {exc}") from exc
    _require_finite_json(payload, path=str(path))
    return payload


def _require_finite_json(value: object, *, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require(np.isfinite(float(value)), f"nonfinite JSON number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_json(item, path=f"{path}.{key}")
        return
    _fail(f"unsupported JSON value at {path}: {type(value).__name__}")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    _require(path.is_file(), f"required CSV is missing: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames is not None, f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CampaignQAFailure(f"cannot read CSV {path}: {exc}") from exc


def _strict_bool(value: object, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    _fail(f"{label} is not a strict boolean: {value!r}")


def _float(value: object, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignQAFailure(f"{label} is not numeric: {value!r}") from exc
    _require(np.isfinite(result), f"{label} is nonfinite")
    return result


def _integer(value: object, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CampaignQAFailure(f"{label} is not an integer: {value!r}") from exc
    return result


def _close(
    actual: object,
    expected: object,
    *,
    label: str,
    rtol: float = 2.0e-11,
    atol: float = 1.0e-13,
) -> float:
    left = _float(actual, label=f"{label} actual")
    right = _float(expected, label=f"{label} expected")
    residual = abs(left - right)
    _require(
        bool(np.isclose(left, right, rtol=rtol, atol=atol)),
        f"{label} mismatch: actual={left:.17g}, expected={right:.17g}",
    )
    return residual


def _exact_float_grid(actual: Sequence[object], expected: Sequence[float], *, label: str) -> None:
    parsed = np.asarray([_float(value, label=label) for value in actual], dtype=float)
    target = np.asarray(expected, dtype=float)
    _require(parsed.shape == target.shape, f"{label} has shape {parsed.shape}, expected {target.shape}")
    _require(np.array_equal(parsed, target), f"{label} does not equal the exact requested grid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_path(value: object) -> str:
    return str(value).replace("\\", "/").rstrip("/")


def _require_path_suffix(value: object, suffix: Path, *, label: str) -> None:
    normalized = _normalized_path(value)
    expected = suffix.as_posix().rstrip("/")
    _require(
        normalized.endswith(expected),
        f"{label} path {normalized!r} does not end with {expected!r}",
    )
    _require("mot_multilevel" not in normalized.lower(), f"{label} points into mot_multilevel")


def _png_dimensions(path: Path) -> tuple[int, int]:
    _require(path.is_file(), f"required final figure is missing: {path}")
    _require(path.stat().st_size >= 25_000, f"final figure is unexpectedly small: {path}")
    try:
        with Image.open(path) as image:
            _require(image.format == "PNG", f"final figure is not PNG encoded: {path}")
            width, height = image.size
            image.verify()
    except CampaignQAFailure:
        raise
    except (OSError, SyntaxError, ValueError) as exc:
        raise CampaignQAFailure(f"final PNG is corrupt or truncated: {path}: {exc}") from exc
    _require(width >= 1600 and height >= 1000, f"final figure resolution is too low ({width}x{height}): {path}")
    return width, height


def _search_from_metadata(metadata: Mapping[str, object]) -> CaptureSearchConfig:
    raw = metadata.get("search_config")
    _require(isinstance(raw, Mapping), "campaign search_config is missing or malformed")
    try:
        search = CaptureSearchConfig(**dict(raw))
    except (TypeError, ValueError) as exc:
        raise CampaignQAFailure(f"campaign search_config cannot be reconstructed: {exc}") from exc
    expected = campaign.default_search_config()
    _require(asdict(search) == asdict(expected), "campaign search_config differs from the canonical 25x25 configuration")
    _require(search.disc_count == EXPECTED_DISC_COUNT, "disc_count is not 25")
    _require(search.points_per_disc == EXPECTED_POINTS_PER_DISC, "points_per_disc is not 25")
    _close(search.disc_radius_m, EXPECTED_DISC_RADIUS_M, label="sampling-disc radius", rtol=0.0)
    _require(search.seed == EXPECTED_SEED, "common-geometry seed is not 20260903")
    _require(not search.include_center_point, "geometry contains a forced disc-center sample")
    _exact_float_grid(
        np.arange(
            search.analysis_velocity_min_m_per_s,
            search.analysis_velocity_max_m_per_s
            + 0.5 * search.analysis_velocity_step_m_per_s,
            search.analysis_velocity_step_m_per_s,
        ),
        EXPECTED_VELOCITY_GRID,
        label="analysis velocity grid",
    )
    return search


def _validate_campaign_header(
    statistics_dir: Path,
) -> tuple[Mapping[str, object], CaptureSearchConfig, str, tuple[object, ...]]:
    raw_metadata = _read_json(statistics_dir / "campaign_metadata.json")
    _require(isinstance(raw_metadata, Mapping), "campaign metadata is not a JSON object")
    metadata = raw_metadata
    _require(
        _integer(metadata.get("schema_version"), label="campaign schema version")
        == campaign.CAMPAIGN_SCHEMA_VERSION,
        "campaign schema version differs from the active source contract",
    )
    _require(metadata.get("status") == "completed", "campaign status is not completed")
    _require(not metadata.get("last_error"), "completed campaign retains a nonempty last_error")
    _require(metadata.get("campaign_name") == campaign.CAMPAIGN_NAME, "campaign name does not match the active source contract")
    _require(statistics_dir.name == campaign.CAMPAIGN_NAME, "statistics-root name does not match the active campaign name")
    _require(metadata.get("execution_order") == list(campaign.STUDY_ORDER), "loading studies were not recorded in canonical sequential order")
    stage_status = metadata.get("stage_status")
    _require(isinstance(stage_status, Mapping), "campaign stage_status is missing")
    _require(
        all(stage_status.get(key) == "completed" for key in campaign.STUDY_ORDER),
        "one or more loading stages are not completed",
    )
    _require(_integer(metadata.get("loading_point_count"), label="loading_point_count") == EXPECTED_LOADING_POINT_COUNT, "campaign loading-point count is not 67")
    _require(
        _integer(metadata.get("capture_threshold_search_count"), label="capture_threshold_search_count")
        == EXPECTED_TOTAL_RAY_COUNT,
        "campaign trajectory count is not 41,875",
    )
    _require(metadata.get("phase_space") == "full_sphere", "campaign is not full-sphere")
    _close(metadata.get("cooling_reference_power_w_per_beam"), EXPECTED_REFERENCE_POWER_W, label="reference cooling power", rtol=0.0)
    temperature_stage = str(metadata.get("temperature_stage", "")).lower()
    _require("omitted" in temperature_stage, "campaign does not explicitly record the temperature stage as omitted")

    grids = metadata.get("requested_grids")
    _require(isinstance(grids, Mapping), "campaign requested_grids is missing")
    _exact_float_grid(grids.get("s0", ()), EXPECTED_RAW_SATURATION, label="raw-saturation campaign grid")
    _exact_float_grid(grids.get("s_eff", ()), EXPECTED_EFFECTIVE_SATURATION, label="effective-saturation campaign grid")
    _exact_float_grid(grids.get("detuning_delta_over_gamma", ()), EXPECTED_LOADING_DETUNING_N, label="loading-detuning campaign grid")

    search = _search_from_metadata(metadata)
    discs, points = campaign.generate_common_geometry(search)
    generated_text = geometry_csv_text(geometry_rows(discs, points))
    generated_hash = hashlib.sha256(generated_text.encode("utf-8")).hexdigest()
    _require(generated_hash == EXPECTED_GEOMETRY_SHA256, "seeded geometry regeneration changed")
    geometry_path = statistics_dir / "launch_geometry.csv"
    _require(geometry_path.is_file(), "campaign launch_geometry.csv is missing")
    saved_hash = _sha256(geometry_path)
    _require(saved_hash == generated_hash, "saved common geometry differs byte-for-byte from seeded regeneration")
    _require(metadata.get("common_geometry_sha256") == saved_hash, "campaign geometry hash is inconsistent")

    directions = np.asarray([disc.incident_unit_vector for disc in discs], dtype=float)
    _require(directions.shape == (25, 3), "common geometry does not contain 25 directions")
    _require(np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0.0, atol=1.0e-13), "incident directions are not unit vectors")
    for axis, label in enumerate("xyz"):
        _require(np.min(directions[:, axis]) < 0.0 < np.max(directions[:, axis]), f"full-sphere {label} directions do not span both signs")
    radii = np.asarray([point.s_m for point in points], dtype=float)
    _require(np.all((radii >= 0.0) & (radii <= EXPECTED_DISC_RADIUS_M)), "disc samples fall outside the 15 mm sampling disc")
    _require(len(points) == EXPECTED_RAY_COUNT_PER_POINT, "seeded common geometry does not contain 625 points")

    expected_campaign_signature = campaign._campaign_signature(search, saved_hash)
    _require(
        metadata.get("campaign_signature_sha256") == expected_campaign_signature,
        "campaign signature does not match current source, grids, and geometry",
    )
    return metadata, search, saved_hash, tuple(points)


def _json_semantically_equal(actual: object, expected: object) -> bool:
    """Compare JSON values exactly while treating tuples as JSON arrays."""

    try:
        actual_text = json.dumps(
            actual, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        expected_text = json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return False
    return actual_text == expected_text


def _mapping_equal(actual: object, expected: object, *, label: str) -> None:
    _require(
        _json_semantically_equal(actual, expected),
        f"{label} does not match the exact source-derived contract",
    )


def _validate_signature_payload(
    actual: object,
    expected: Mapping[str, object],
    expected_signature: str,
    *,
    label: str,
) -> None:
    """Tie a JSON-restored payload to the exact canonical source signature."""

    _require(isinstance(actual, Mapping), f"{label} is malformed")
    _require(
        campaign._signature(actual) == expected_signature,
        f"{label} canonical signature differs from the source-derived contract",
    )
    _mapping_equal(actual, expected, label=label)


def _validate_parameter_contract(
    point: campaign.RelationshipPoint,
    metadata: Mapping[str, object],
) -> None:
    relationship = metadata.get("relationship_point")
    _mapping_equal(relationship, asdict(point), label=f"{point.slug} relationship_point")
    apparatus = metadata.get("apparatus_config")
    simple = metadata.get("simple_mot_config")
    _require(isinstance(apparatus, Mapping) and isinstance(simple, Mapping), f"{point.slug} configuration metadata is malformed")
    cooling = apparatus.get("cooling")
    _require(isinstance(cooling, Mapping), f"{point.slug} cooling configuration is malformed")
    _close(cooling.get("power_w_per_beam"), point.cooling_power_w_per_beam, label=f"{point.slug} cooling power")
    _close(cooling.get("detuning_hz"), point.cooling_detuning_hz, label=f"{point.slug} cooling detuning")
    _close(cooling.get("beam_diameter_m"), EXPECTED_BEAM_DIAMETER_M, label=f"{point.slug} beam diameter", rtol=0.0)
    _close(simple.get("cooling_detuning_hz"), point.cooling_detuning_hz, label=f"{point.slug} simple detuning")
    _require(metadata.get("repumper_included") is False, f"{point.slug} unexpectedly includes a repumper")
    _require(_integer(metadata.get("built_cooling_beam_count"), label=f"{point.slug} built beam count") == 6, f"{point.slug} does not contain six cooling beams")
    powers = metadata.get("built_cooling_beam_powers_w")
    detunings = metadata.get("built_cooling_beam_detunings_hz")
    _require(isinstance(powers, list) and len(powers) == 6, f"{point.slug} built beam powers are malformed")
    _require(isinstance(detunings, list) and len(detunings) == 6, f"{point.slug} built beam detunings are malformed")
    for index, value in enumerate(powers):
        _close(value, point.cooling_power_w_per_beam, label=f"{point.slug} beam {index} power")
    for index, value in enumerate(detunings):
        _close(value, point.cooling_detuning_hz, label=f"{point.slug} beam {index} detuning")
    _require("with gravity" in str(metadata.get("capture_dynamics", "")).lower(), f"{point.slug} capture dynamics do not state that gravity is enabled")

    if point.study_key in {campaign.RAW_STUDY_KEY, campaign.EFFECTIVE_STUDY_KEY}:
        _close(point.cooling_detuning_hz, -15.0e6, label=f"{point.slug} fixed saturation-sweep detuning", rtol=0.0)
    elif point.study_key == campaign.DETUNING_STUDY_KEY:
        _close(point.cooling_power_w_per_beam, EXPECTED_REFERENCE_POWER_W, label=f"{point.slug} fixed detuning-sweep power", rtol=0.0)
    else:  # pragma: no cover - guarded by source-derived plans
        _fail(f"unexpected study key: {point.study_key}")

    diameter = _float(cooling.get("beam_diameter_m"), label=f"{point.slug} beam diameter")
    peak_intensity = 2.0 * point.cooling_power_w_per_beam / (pi * (0.5 * diameter) ** 2)
    saturation_intensity = _float(simple.get("saturation_intensity_w_per_m2"), label=f"{point.slug} saturation intensity")
    reconstructed_s0 = peak_intensity / saturation_intensity
    reconstructed_seff = reconstructed_s0 / (1.0 + (2.0 * point.cooling_detuning_n) ** 2)
    _close(reconstructed_s0, point.on_resonance_saturation, label=f"{point.slug} reconstructed s0", rtol=3.0e-13, atol=0.0)
    _close(reconstructed_seff, point.effective_saturation, label=f"{point.slug} reconstructed seff", rtol=3.0e-13, atol=0.0)


def _compare_numeric_row(
    saved: Mapping[str, object],
    expected: Mapping[str, object],
    numeric_fields: Iterable[str],
    *,
    label: str,
) -> float:
    maximum = 0.0
    for field in numeric_fields:
        maximum = max(maximum, _close(saved.get(field), expected[field], label=f"{label}.{field}"))
    return maximum


def _reconstruct_loading(
    samples: Sequence[CaptureVelocitySample],
    overrides_payload: Mapping[str, object],
) -> tuple[
    list[dict[str, float | int]],
    list[dict[str, float | int]],
    dict[str, float | int | bool],
]:
    ordered = sorted(samples, key=lambda sample: (sample.disc_index, sample.point_index))
    _require(len(ordered) == EXPECTED_RAY_COUNT_PER_POINT, "point does not contain 625 ordered samples")
    expected_keys = [(disc, point) for disc in range(25) for point in range(25)]
    actual_keys = [(sample.disc_index, sample.point_index) for sample in ordered]
    _require(actual_keys == expected_keys, "sample keys are not the exact 25x25 Cartesian index set")

    thresholds = np.asarray([sample.capture_velocity_m_per_s for sample in ordered], dtype=float)
    lower_trapped = np.asarray([sample.lower_classification in TRAPPED_REASONS for sample in ordered], dtype=bool)
    captured = lower_trapped[:, None] & (
        thresholds[:, None] >= EXPECTED_VELOCITY_GRID[None, :] - 1.0e-12
    )
    indeterminate = np.zeros(captured.shape, dtype=bool)
    overrides = velocity_overrides_from_payload(overrides_payload)
    seen: set[tuple[int, int]] = set()
    indeterminate_override_keys: set[tuple[int, int]] = set()
    for override in overrides:
        key = (override.disc_index, override.point_index)
        _require(key not in seen, f"duplicate override {key}")
        _require(key in expected_keys, f"override {key} has no canonical sample ray")
        seen.add(key)
        _exact_float_grid(override.velocity_m_per_s, EXPECTED_VELOCITY_GRID, label=f"override {key} velocity grid")
        _require(
            len(override.captured) == len(EXPECTED_VELOCITY_GRID),
            f"override {key} mask does not contain 121 states",
        )
        _require(
            all(value is None or type(value) is bool for value in override.captured),
            f"override {key} mask contains a non-boolean/non-null state",
        )
        none_indices = [
            index for index, value in enumerate(override.captured) if value is None
        ]
        _require(
            not none_indices or none_indices == [0],
            f"override {key} is indeterminate away from exact v=0",
        )
        _require(
            override.captured[-1] is False,
            f"override {key} captures the 30 m/s upper endpoint",
        )
        flat_index = override.disc_index * EXPECTED_POINTS_PER_DISC + override.point_index
        captured[flat_index, :] = np.asarray(
            [False if value is None else value for value in override.captured],
            dtype=bool,
        )
        if none_indices:
            indeterminate[flat_index, 0] = True
            indeterminate_override_keys.add(key)

    indeterminate_sample_keys = {
        (sample.disc_index, sample.point_index)
        for sample in ordered
        if sample.lower_classification
        == campaign.INDETERMINATE_ZERO_FLUX_CLASSIFICATION
    }
    _require(
        indeterminate_override_keys == indeterminate_sample_keys,
        "indeterminate zero-flux scalar sentinels and direct masks disagree",
    )
    _require(
        not np.any(indeterminate[:, 1:]),
        "an indeterminate capture state occurs at positive loading flux",
    )
    indeterminate_zero_flux_ray_count = len(indeterminate_override_keys)

    area = pi * EXPECTED_DISC_RADIUS_M**2
    disc_masks = captured.reshape(EXPECTED_DISC_COUNT, EXPECTED_POINTS_PER_DISC, -1)
    disc_sigma = area * np.mean(disc_masks, axis=1)
    mean_sigma = np.clip(np.mean(disc_sigma, axis=0), 0.0, area)
    sample_std_sigma = np.std(disc_sigma, axis=0, ddof=1)
    sem_sigma = sample_std_sigma / np.sqrt(EXPECTED_DISC_COUNT)
    half_sigma = EXPECTED_T_CRITICAL_95_DF24 * sem_sigma
    lower_sigma = np.clip(mean_sigma - half_sigma, 0.0, mean_sigma)
    upper_sigma = np.clip(mean_sigma + half_sigma, mean_sigma, area)
    spectrum: list[dict[str, float | int]] = []
    first_spectrum_index = 1 if indeterminate_zero_flux_ray_count else 0
    for index, speed in enumerate(
        EXPECTED_VELOCITY_GRID[first_spectrum_index:],
        start=first_spectrum_index,
    ):
        spectrum.append(
            {
                "velocity_m_per_s": float(speed),
                "captured_count": int(np.count_nonzero(captured[:, index])),
                "launched_count": EXPECTED_RAY_COUNT_PER_POINT,
                "capture_fraction": float(mean_sigma[index] / area),
                "capture_cross_section_m2": float(mean_sigma[index]),
                "capture_cross_section_sample_std_m2": float(sample_std_sigma[index]),
                "capture_cross_section_disc_cluster_sem_m2": float(sem_sigma[index]),
                "capture_cross_section_t95_lower_m2": float(lower_sigma[index]),
                "capture_cross_section_t95_upper_m2": float(upper_sigma[index]),
                "disc_count": EXPECTED_DISC_COUNT,
                "student_t_critical_95": EXPECTED_T_CRITICAL_95_DF24,
            }
        )

    positive_velocity = EXPECTED_VELOCITY_GRID[1:]
    positive_weights = positive_velocity**3 * np.exp(
        -(positive_velocity**2) / THERMAL_SCALE_M2_PER_S2
    )
    if indeterminate_zero_flux_ray_count:
        # The v=0 cross section is never reconstructed.  Only the exact
        # algebraic endpoint of the weighted integrand, g(0)=0, is supplied.
        disc_integrand = np.concatenate(
            (
                np.zeros((EXPECTED_DISC_COUNT, 1), dtype=float),
                disc_sigma[:, 1:] * positive_weights[None, :],
            ),
            axis=1,
        )
        mean_integrand = np.concatenate(
            (
                np.asarray([0.0]),
                mean_sigma[1:] * positive_weights,
            )
        )
    else:
        weights = EXPECTED_VELOCITY_GRID**3 * np.exp(
            -(EXPECTED_VELOCITY_GRID**2) / THERMAL_SCALE_M2_PER_S2
        )
        disc_integrand = disc_sigma * weights[None, :]
        mean_integrand = mean_sigma * weights
    integrals = np.trapezoid(
        disc_integrand, EXPECTED_VELOCITY_GRID, axis=1
    )
    rates = LOADING_RATE_PREFACTOR * integrals
    loading_by_disc = [
        {
            "disc_index": index,
            "point_count": EXPECTED_POINTS_PER_DISC,
            "loading_integral_m6_per_s4": float(integrals[index]),
            "loading_rate_atoms_per_s": float(rates[index]),
        }
        for index in range(EXPECTED_DISC_COUNT)
    ]
    mean_integral = float(np.mean(integrals))
    mean_spectrum_integral = float(
        np.trapezoid(mean_integrand, EXPECTED_VELOCITY_GRID)
    )
    mean_rate = float(np.mean(rates))
    spectrum_rate = LOADING_RATE_PREFACTOR * mean_spectrum_integral
    sample_std_rate = float(np.std(rates, ddof=1))
    sem_rate = sample_std_rate / np.sqrt(EXPECTED_DISC_COUNT)
    half_rate = EXPECTED_T_CRITICAL_95_DF24 * sem_rate
    loading = {
        "loading_rate_mean_atoms_per_s": mean_rate,
        "loading_rate_from_mean_spectrum_atoms_per_s": spectrum_rate,
        "loading_rate_sample_std_atoms_per_s": sample_std_rate,
        "loading_rate_disc_cluster_sem_atoms_per_s": sem_rate,
        "loading_rate_t95_lower_atoms_per_s": max(0.0, mean_rate - half_rate),
        "loading_rate_t95_upper_atoms_per_s": mean_rate + half_rate,
        "student_t_critical_95": EXPECTED_T_CRITICAL_95_DF24,
        "confidence_level": EXPECTED_CONFIDENCE,
        "disc_count": EXPECTED_DISC_COUNT,
        "point_count": EXPECTED_RAY_COUNT_PER_POINT,
        "loading_integral_mean_m6_per_s4": mean_integral,
        "loading_integral_from_mean_spectrum_m6_per_s4": mean_spectrum_integral,
        "velocity_min_m_per_s": 0.0,
        "velocity_max_m_per_s": 30.0,
        "velocity_grid_sample_count": len(EXPECTED_VELOCITY_GRID),
        "capture_spectrum_row_count": len(spectrum),
        "indeterminate_zero_flux_ray_count": indeterminate_zero_flux_ray_count,
        "zero_flux_quadrature_anchor_used": bool(
            indeterminate_zero_flux_ray_count
        ),
        "zero_speed_cross_section_imputed": False,
    }
    return spectrum, loading_by_disc, loading


def _reconstruct_audit_counts(
    audit_rows: Mapping[tuple[int, int], Mapping[str, object]],
    override_count: int,
) -> dict[str, int]:
    """Independently derive every point-level audit counter from saved rows."""

    return {
        "automatically_extended_timeout_ray_count": sum(
            _strict_bool(
                row.get("base_timeout_detected"),
                label=f"audit {key} base_timeout_detected",
            )
            for key, row in audit_rows.items()
        ),
        "adaptively_extended_velocity_grid_ray_count": sum(
            _integer(
                row.get("adaptive_audit_level_count"),
                label=f"audit {key} adaptive level count",
            )
            > 0
            and str(row.get("zero_threshold_audit_status")) != "not_applicable"
            for key, row in audit_rows.items()
        ),
        "adaptively_extended_audit_ray_count": sum(
            _integer(
                row.get("adaptive_audit_level_count"),
                label=f"audit {key} adaptive level count",
            )
            > 0
            for key, row in audit_rows.items()
        ),
        "adaptively_extended_positive_boundary_ray_count": sum(
            str(row.get("timeout_resolution_status"))
            == "adaptive_dual_step_research_recovered_boundary"
            for row in audit_rows.values()
        ),
        "complete_longer_boundary_fallback_ray_count": sum(
            str(row.get("timeout_resolution_status"))
            == "complete_scalar_diagnostics_and_velocity_grid_recovered_capture"
            for row in audit_rows.values()
        ),
        "positive_boundary_grid_override_count": sum(
            str(row.get("positive_boundary_grid_audit_status"))
            in {
                "velocity_resolved_capture",
                campaign.INDETERMINATE_ZERO_FLUX_STATUS,
            }
            for row in audit_rows.values()
        ),
        "pre_adaptive_positive_research_timeout_ray_count": sum(
            _integer(
                row.get("pre_adaptive_coarse_evaluation_timeout_count"),
                label=f"audit {key} pre-adaptive coarse timeout count",
            )
            + _integer(
                row.get("pre_adaptive_fine_evaluation_timeout_count"),
                label=f"audit {key} pre-adaptive fine timeout count",
            )
            > 0
            for key, row in audit_rows.items()
        ),
        "velocity_resolved_override_count": override_count,
        "indeterminate_zero_flux_ray_count": sum(
            str(row.get("zero_threshold_audit_status"))
            == campaign.INDETERMINATE_ZERO_FLUX_STATUS
            or str(row.get("positive_boundary_grid_audit_status"))
            == campaign.INDETERMINATE_ZERO_FLUX_STATUS
            for row in audit_rows.values()
        ),
    }


def _validate_point(
    point: campaign.RelationshipPoint,
    *,
    search: CaptureSearchConfig,
    geometry_hash: str,
    common_points: Sequence[object],
    statistics_dir: Path,
    figures_dir: Path,
    aggregate_row: Mapping[str, str],
) -> PointQAResult:
    point_statistics = statistics_dir / point.study_key / "points" / point.slug
    point_figures = figures_dir / point.study_key / "points" / point.slug
    metadata_payload = _read_json(point_statistics / "run_metadata.json")
    _require(isinstance(metadata_payload, Mapping), f"{point.slug} run metadata is malformed")
    metadata = metadata_payload
    _require(
        _integer(metadata.get("schema_version"), label=f"{point.slug} schema version")
        == campaign.POINT_SCHEMA_VERSION,
        f"{point.slug} schema version differs from the active source contract",
    )
    _require(metadata.get("status") == "completed", f"{point.slug} status is not completed")
    _require(not metadata.get("last_error"), f"{point.slug} retains a nonempty last_error")
    _require(_integer(metadata.get("completed_sample_count"), label=f"{point.slug} sample count") == EXPECTED_RAY_COUNT_PER_POINT, f"{point.slug} metadata does not record 625 samples")
    _close(metadata.get("completion_fraction"), 1.0, label=f"{point.slug} completion fraction", rtol=0.0)
    _require(metadata.get("phase_space") == "full_sphere", f"{point.slug} is not full-sphere")
    _require(metadata.get("geometry_sha256") == geometry_hash, f"{point.slug} geometry hash differs")
    _validate_parameter_contract(point, metadata)

    expected_signature_payload = campaign._point_signature_payload(point, search, geometry_hash)
    expected_signature = campaign._signature(expected_signature_payload)
    _validate_signature_payload(
        metadata.get("signature_payload"),
        expected_signature_payload,
        expected_signature,
        label=f"{point.slug} signature payload",
    )
    _mapping_equal(
        metadata.get("endpoint_timeout_policy"),
        expected_signature_payload.get("endpoint_timeout_policy"),
        label=f"{point.slug} endpoint timeout policy",
    )
    _require(metadata.get("run_signature_sha256") == expected_signature, f"{point.slug} run signature is invalid")
    actual_sources = campaign._source_hashes()
    _mapping_equal(expected_signature_payload.get("physics_source_sha256"), actual_sources, label=f"{point.slug} physics source hashes")

    geometry_path = point_statistics / "launch_geometry.csv"
    _require(geometry_path.is_file(), f"{point.slug} point geometry is missing")
    _require(_sha256(geometry_path) == geometry_hash, f"{point.slug} point geometry differs from common geometry")
    final_samples_path = point_statistics / "capture_velocity_samples.csv"
    partial_samples_path = point_statistics / "capture_velocity_partial_samples.csv"
    _require(final_samples_path.is_file(), f"{point.slug} final sample ledger is missing")
    _require(partial_samples_path.is_file(), f"{point.slug} partial sample checkpoint is missing")
    _require(_sha256(final_samples_path) == _sha256(partial_samples_path), f"{point.slug} final and partial sample ledgers differ")
    sample_header, raw_sample_rows = _read_csv(final_samples_path)
    _require(sample_header == list(SAMPLE_FIELDNAMES), f"{point.slug} sample schema differs")
    _require(len(raw_sample_rows) == EXPECTED_RAY_COUNT_PER_POINT, f"{point.slug} raw sample ledger does not contain 625 rows")
    for raw_index, raw_row in enumerate(raw_sample_rows):
        _strict_bool(raw_row.get("lower_entered_trap_core"), label=f"{point.slug} sample[{raw_index}] lower_entered_trap_core")
        _strict_bool(raw_row.get("upper_entered_trap_core"), label=f"{point.slug} sample[{raw_index}] upper_entered_trap_core")
    samples = load_capture_velocity_samples(final_samples_path)
    sample_map = validate_checkpoint_samples(samples, common_points)
    _require(len(sample_map) == EXPECTED_RAY_COUNT_PER_POINT, f"{point.slug} sample ledger does not contain 625 unique rays")
    for key, sample in sample_map.items():
        numeric = np.asarray(
            [
                sample.capture_velocity_m_per_s,
                sample.velocity_resolution_m_per_s,
                sample.trapped_velocity_lower_m_per_s,
                sample.untrapped_velocity_upper_m_per_s,
                *sample.initial_position_m,
                *sample.incident_unit_vector,
            ],
            dtype=float,
        )
        _require(np.all(np.isfinite(numeric)), f"{point.slug} sample {key} contains nonfinite data")
        _require(sample.upper_classification == "escaped", f"{point.slug} sample {key} upper endpoint is not escaped")
        _require(sample.lower_classification not in UNRESOLVED_REASONS, f"{point.slug} sample {key} has unresolved lower endpoint")
        _require(sample.upper_classification not in UNRESOLVED_REASONS, f"{point.slug} sample {key} has unresolved upper endpoint")
        if sample.capture_velocity_m_per_s > 0.0:
            _require(sample.lower_classification in TRAPPED_REASONS, f"{point.slug} sample {key} positive lower endpoint is not trapped")

    audit_path = point_statistics / "capture_endpoint_audit.csv"
    audit_header, audit_rows_list = _read_csv(audit_path)
    _require(audit_header == list(campaign.ENDPOINT_AUDIT_FIELDNAMES), f"{point.slug} audit schema differs from active source")
    audit_rows: dict[tuple[int, int], dict[str, object]] = {}
    for row in audit_rows_list:
        key = (
            _integer(row.get("disc_index"), label=f"{point.slug} audit disc index"),
            _integer(row.get("point_index"), label=f"{point.slug} audit point index"),
        )
        _require(key not in audit_rows, f"{point.slug} duplicate audit row {key}")
        audit_rows[key] = dict(row)
        _require(str(row.get("accepted_lower_classification")) not in UNRESOLVED_REASONS, f"{point.slug} audit {key} accepted lower endpoint is unresolved")
        _require(str(row.get("accepted_upper_classification")) == "escaped", f"{point.slug} audit {key} accepted upper endpoint is not escaped")
    _require(len(audit_rows) == EXPECTED_RAY_COUNT_PER_POINT, f"{point.slug} audit ledger does not contain 625 rows")

    overrides_payload = _read_json(point_statistics / "capture_velocity_overrides.json")
    _require(isinstance(overrides_payload, Mapping), f"{point.slug} override ledger is malformed")
    overrides = velocity_overrides_from_payload(overrides_payload)
    overrides_by_key = {override.key: override for override in overrides}
    _require(len(overrides_by_key) == len(overrides), f"{point.slug} has duplicate override keys")
    try:
        campaign._validate_completed_audit_ledger(
            sample_map,
            audit_rows,
            overrides_by_key,
            search=search,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignQAFailure(f"{point.slug} fail-closed audit ledger validation failed: {exc}") from exc
    audit_counts = _reconstruct_audit_counts(audit_rows, len(overrides))
    for field, expected_count in audit_counts.items():
        _require(
            _integer(metadata.get(field), label=f"{point.slug} metadata {field}")
            == expected_count,
            f"{point.slug} metadata {field} differs from its audit ledger",
        )

    expected_spectrum, expected_by_disc, expected_loading = _reconstruct_loading(
        samples, overrides_payload
    )
    spectrum_header, saved_spectrum = _read_csv(point_statistics / "capture_velocity_spectrum.csv")
    _require(spectrum_header == list(campaign.SPECTRUM_FIELDNAMES), f"{point.slug} spectrum schema differs")
    _require(
        len(saved_spectrum) == len(expected_spectrum),
        f"{point.slug} spectrum row count does not match its tri-state evidence",
    )
    spectrum_numeric = tuple(expected_spectrum[0])
    maximum_residual = 0.0
    for index, (saved, expected) in enumerate(zip(saved_spectrum, expected_spectrum, strict=True)):
        maximum_residual = max(
            maximum_residual,
            _compare_numeric_row(saved, expected, spectrum_numeric, label=f"{point.slug} spectrum[{index}]"),
        )

    by_disc_header, saved_by_disc = _read_csv(point_statistics / "loading_rate_by_disc.csv")
    _require(by_disc_header == list(campaign.LOADING_BY_DISC_FIELDNAMES), f"{point.slug} per-disc loading schema differs")
    _require(len(saved_by_disc) == EXPECTED_DISC_COUNT, f"{point.slug} does not contain 25 disc loading estimates")
    by_disc_numeric = tuple(expected_by_disc[0])
    for index, (saved, expected) in enumerate(zip(saved_by_disc, expected_by_disc, strict=True)):
        maximum_residual = max(
            maximum_residual,
            _compare_numeric_row(saved, expected, by_disc_numeric, label=f"{point.slug} loading_by_disc[{index}]"),
        )

    loading_payload = _read_json(point_statistics / "loading_rate_result.json")
    summary_payload = _read_json(point_statistics / "capture_velocity_summary.json")
    _require(isinstance(loading_payload, Mapping) and isinstance(summary_payload, Mapping), f"{point.slug} analysis JSON is malformed")
    loading_numeric = tuple(
        key
        for key, value in expected_loading.items()
        if isinstance(value, (int, float)) and type(value) is not bool
    )
    maximum_residual = max(
        maximum_residual,
        _compare_numeric_row(loading_payload, expected_loading, loading_numeric, label=f"{point.slug} loading"),
    )
    _require(loading_payload.get("run_signature_sha256") == expected_signature, f"{point.slug} loading JSON signature differs")
    _require(summary_payload.get("run_signature_sha256") == expected_signature, f"{point.slug} summary JSON signature differs")
    _require(summary_payload.get("geometry_sha256") == geometry_hash, f"{point.slug} summary geometry hash differs")
    _mapping_equal(summary_payload.get("loading_rate"), loading_payload, label=f"{point.slug} summary/loading JSON duplication")
    _mapping_equal(metadata.get("loading_rate"), loading_payload, label=f"{point.slug} metadata/loading JSON duplication")
    _require(_integer(summary_payload.get("unresolved_timeout_count"), label=f"{point.slug} unresolved count") == 0, f"{point.slug} summary reports unresolved timeouts")
    _require(_integer(summary_payload.get("sample_count"), label=f"{point.slug} summary sample count") == EXPECTED_RAY_COUNT_PER_POINT, f"{point.slug} summary sample count differs")
    _require(_integer(summary_payload.get("velocity_resolved_override_count"), label=f"{point.slug} summary override count") == len(overrides), f"{point.slug} summary override count differs")
    _require(_integer(metadata.get("velocity_resolved_override_count"), label=f"{point.slug} metadata override count") == len(overrides), f"{point.slug} metadata override count differs")
    indeterminate_count = _integer(
        expected_loading["indeterminate_zero_flux_ray_count"],
        label=f"{point.slug} reconstructed indeterminate-zero count",
    )
    for owner, payload in (
        ("metadata", metadata),
        ("summary", summary_payload),
        ("loading", loading_payload),
    ):
        _require(
            _integer(
                payload.get("indeterminate_zero_flux_ray_count"),
                label=f"{point.slug} {owner} indeterminate-zero count",
            )
            == indeterminate_count,
            f"{point.slug} {owner} indeterminate-zero count differs",
        )
    _require(
        _integer(
            loading_payload.get("capture_spectrum_row_count"),
            label=f"{point.slug} loading spectrum-row count",
        )
        == len(expected_spectrum),
        f"{point.slug} loading spectrum-row count differs",
    )
    _require(
        _integer(
            loading_payload.get("velocity_grid_sample_count"),
            label=f"{point.slug} loading quadrature-grid count",
        )
        == len(EXPECTED_VELOCITY_GRID),
        f"{point.slug} loading quadrature grid lost its v=0 endpoint",
    )
    _require(
        type(loading_payload.get("zero_flux_quadrature_anchor_used")) is bool
        and loading_payload.get("zero_flux_quadrature_anchor_used")
        is bool(indeterminate_count),
        f"{point.slug} loading zero-flux anchor flag differs",
    )
    _require(
        loading_payload.get("zero_speed_cross_section_imputed") is False,
        f"{point.slug} loading claims or omits the no-imputation invariant",
    )

    capture = np.asarray([sample.capture_velocity_m_per_s for sample in samples], dtype=float)
    summary_expected = {
        "capture_velocity_mean_m_per_s": float(np.mean(capture)),
        "capture_velocity_sample_std_m_per_s": float(np.std(capture, ddof=1)),
        "capture_velocity_min_m_per_s": float(np.min(capture)),
        "capture_velocity_max_m_per_s": float(np.max(capture)),
    }
    maximum_residual = max(
        maximum_residual,
        _compare_numeric_row(summary_payload, summary_expected, summary_expected, label=f"{point.slug} capture summary"),
    )
    _require(_integer(summary_payload.get("zero_capture_velocity_count"), label=f"{point.slug} zero count") == int(np.count_nonzero(capture == 0.0)), f"{point.slug} zero-threshold count differs")
    lower_counts = dict(sorted(Counter(sample.lower_classification for sample in samples).items()))
    upper_counts = dict(sorted(Counter(sample.upper_classification for sample in samples).items()))
    _mapping_equal(summary_payload.get("lower_classification_counts"), lower_counts, label=f"{point.slug} lower classification counts")
    _mapping_equal(summary_payload.get("upper_classification_counts"), upper_counts, label=f"{point.slug} upper classification counts")
    _close(loading_payload.get("loading_rate_prefactor"), LOADING_RATE_PREFACTOR, label=f"{point.slug} loading prefactor", rtol=0.0)
    _close(loading_payload.get("thermal_scale_m2_per_s2"), THERMAL_SCALE_M2_PER_S2, label=f"{point.slug} thermal scale", rtol=0.0)
    _require(loading_payload.get("quadrature_method") == "trapezoid", f"{point.slug} loading quadrature is not trapezoidal")
    _require("disc" in str(loading_payload.get("primary_uncertainty", "")).lower(), f"{point.slug} primary uncertainty is not disc-clustered")

    simple_metadata = metadata["simple_mot_config"]
    peak_intensity = 2.0 * point.cooling_power_w_per_beam / (
        pi * (0.5 * EXPECTED_BEAM_DIAMETER_M) ** 2
    )
    aggregate_numeric_expected = {
        "scan_value": point.scan_value,
        "s0": point.on_resonance_saturation,
        "seff": point.effective_saturation,
        "detuning_n": point.cooling_detuning_n,
        "cooling_power_w_per_beam": point.cooling_power_w_per_beam,
        "cooling_power_mw_per_beam": 1.0e3 * point.cooling_power_w_per_beam,
        "cooling_beam_diameter_m": EXPECTED_BEAM_DIAMETER_M,
        "cooling_beam_diameter_mm": 1.0e3 * EXPECTED_BEAM_DIAMETER_M,
        "cooling_beam_center_peak_intensity_w_per_m2": peak_intensity,
        "cooling_beam_center_on_resonance_saturation_parameter": point.on_resonance_saturation,
        "cooling_beam_center_effective_saturation_parameter": point.effective_saturation,
        "cooling_detuning_hz": point.cooling_detuning_hz,
        "cooling_detuning_mhz": point.cooling_detuning_hz / 1.0e6,
        "linewidth_hz": simple_metadata["linewidth_hz"],
        **{
            key: value
            for key, value in expected_loading.items()
            if key.startswith("loading_rate_")
            or key in {"student_t_critical_95", "confidence_level", "disc_count"}
        },
    }
    aggregate_numeric_expected["points_per_disc"] = EXPECTED_POINTS_PER_DISC
    aggregate_numeric_expected["capture_threshold_search_count"] = EXPECTED_RAY_COUNT_PER_POINT
    aggregate_numeric_expected["disc_radius_m"] = EXPECTED_DISC_RADIUS_M
    aggregate_numeric_expected["disc_radius_mm"] = 1.0e3 * EXPECTED_DISC_RADIUS_M
    maximum_residual = max(
        maximum_residual,
        _compare_numeric_row(aggregate_row, aggregate_numeric_expected, aggregate_numeric_expected, label=f"{point.slug} aggregate"),
    )
    _require(_integer(aggregate_row.get("point_index"), label=f"{point.slug} aggregate point index") == point.point_index, f"{point.slug} aggregate point index differs")
    _require(aggregate_row.get("study_key") == point.study_key, f"{point.slug} aggregate study key differs")
    _require(aggregate_row.get("scan_variable") == point.scan_variable, f"{point.slug} aggregate scan variable differs")
    _require(aggregate_row.get("phase_space") == "full_sphere", f"{point.slug} aggregate is not full-sphere")
    _require(aggregate_row.get("geometry_sha256") == geometry_hash, f"{point.slug} aggregate geometry hash differs")
    _require(aggregate_row.get("run_signature_sha256") == expected_signature, f"{point.slug} aggregate signature differs")
    _require(aggregate_row.get("status") == "completed", f"{point.slug} aggregate status differs")
    _require(_integer(aggregate_row.get("unresolved_timeout_count"), label=f"{point.slug} aggregate unresolved count") == 0, f"{point.slug} aggregate reports unresolved timeouts")
    _require(_integer(aggregate_row.get("velocity_resolved_override_count"), label=f"{point.slug} aggregate override count") == len(overrides), f"{point.slug} aggregate override count differs")
    _require(
        _integer(
            aggregate_row.get("indeterminate_zero_flux_ray_count"),
            label=f"{point.slug} aggregate indeterminate-zero count",
        )
        == indeterminate_count,
        f"{point.slug} aggregate indeterminate-zero count differs",
    )
    _require(
        _integer(aggregate_row.get("base_timeout_ray_count"), label=f"{point.slug} aggregate base-timeout count")
        == audit_counts["automatically_extended_timeout_ray_count"],
        f"{point.slug} aggregate base-timeout count differs",
    )
    _require(
        _integer(aggregate_row.get("zero_capture_velocity_count"), label=f"{point.slug} aggregate zero count")
        == int(np.count_nonzero(capture == 0.0)),
        f"{point.slug} aggregate zero-threshold count differs",
    )
    _mapping_equal(
        json.loads(str(aggregate_row.get("lower_classification_counts_json"))),
        lower_counts,
        label=f"{point.slug} aggregate lower classification counts",
    )
    _mapping_equal(
        json.loads(str(aggregate_row.get("upper_classification_counts_json"))),
        upper_counts,
        label=f"{point.slug} aggregate upper classification counts",
    )

    point_identity_relative = Path(point.study_key) / point.slug
    point_output_relative = Path(point.study_key) / "points" / point.slug
    _require_path_suffix(metadata.get("study_name"), Path(campaign.CAMPAIGN_NAME) / point_identity_relative, label=f"{point.slug} study_name")
    _require_path_suffix(aggregate_row.get("statistics_directory"), Path("mot_simple") / campaign.CAMPAIGN_NAME / point_output_relative, label=f"{point.slug} statistics directory")
    _require_path_suffix(aggregate_row.get("figures_directory"), Path("mot_simple") / campaign.CAMPAIGN_NAME / point_output_relative, label=f"{point.slug} figures directory")
    for name in (
        "capture_cross_section_vs_velocity.png",
        "capture_velocity_vs_impact_parameter.png",
        "loading_rate_by_disc.png",
    ):
        _require((point_figures / name).is_file(), f"{point.slug} diagnostic figure is missing: {name}")

    estimator_difference = abs(
        float(expected_loading["loading_rate_mean_atoms_per_s"])
        - float(expected_loading["loading_rate_from_mean_spectrum_atoms_per_s"])
    )
    _require(
        bool(
            np.isclose(
                float(expected_loading["loading_rate_mean_atoms_per_s"]),
                float(expected_loading["loading_rate_from_mean_spectrum_atoms_per_s"]),
                rtol=5.0e-13,
                atol=1.0e-6,
            )
        ),
        f"{point.slug} independently reconstructed loading estimators disagree",
    )
    return PointQAResult(
        study_key=point.study_key,
        point_index=point.point_index,
        ray_count=len(samples),
        audit_row_count=len(audit_rows),
        override_count=len(overrides),
        indeterminate_zero_flux_ray_count=indeterminate_count,
        maximum_numeric_residual=maximum_residual,
        loading_rate_atoms_per_s=float(expected_loading["loading_rate_mean_atoms_per_s"]),
    )


def _validate_study_metadata(
    study_key: str,
    points: Sequence[campaign.RelationshipPoint],
    *,
    statistics_dir: Path,
    figures_dir: Path,
    search: CaptureSearchConfig,
    geometry_hash: str,
) -> list[dict[str, str]]:
    study_dir = statistics_dir / study_key
    expected_point_directories = [point.slug for point in points]
    actual_points_root = study_dir / "points"
    _require(actual_points_root.is_dir(), f"{study_key} points directory is missing")
    actual_point_directories = sorted(
        path.name for path in actual_points_root.iterdir() if path.is_dir()
    )
    _require(
        actual_point_directories == expected_point_directories,
        f"{study_key} point directories do not exactly match the requested plan",
    )
    header, aggregate = _read_csv(study_dir / "aggregate.csv")
    _require(header == list(campaign.AGGREGATE_FIELDNAMES), f"{study_key} aggregate schema differs from active source")
    _require(len(aggregate) == len(points), f"{study_key} aggregate does not contain the exact requested point count")
    indices = [_integer(row.get("point_index"), label=f"{study_key} aggregate point index") for row in aggregate]
    _require(indices == list(range(len(points))), f"{study_key} aggregate point order is not canonical")
    _exact_float_grid([row.get("scan_value") for row in aggregate], [point.scan_value for point in points], label=f"{study_key} aggregate grid")

    metadata_payload = _read_json(study_dir / "sweep_metadata.json")
    _require(isinstance(metadata_payload, Mapping), f"{study_key} sweep metadata is malformed")
    metadata = metadata_payload
    _require(
        _integer(metadata.get("schema_version"), label=f"{study_key} schema version")
        == campaign.CAMPAIGN_SCHEMA_VERSION,
        f"{study_key} schema version differs from the active source contract",
    )
    _require(metadata.get("status") == "completed", f"{study_key} sweep is not completed")
    _require(metadata.get("study_key") == study_key, f"{study_key} sweep metadata key differs")
    _require(_integer(metadata.get("point_count_requested"), label=f"{study_key} requested count") == len(points), f"{study_key} requested point count differs")
    _require(_integer(metadata.get("point_count_completed"), label=f"{study_key} completed count") == len(points), f"{study_key} completed point count differs")
    expected_indeterminate_count = sum(
        _integer(
            row.get("indeterminate_zero_flux_ray_count"),
            label=f"{study_key} aggregate indeterminate-zero count",
        )
        for row in aggregate
    )
    _require(
        _integer(
            metadata.get("indeterminate_zero_flux_ray_count"),
            label=f"{study_key} sweep indeterminate-zero count",
        )
        == expected_indeterminate_count,
        f"{study_key} sweep indeterminate-zero count differs from aggregate",
    )
    _mapping_equal(metadata.get("ordered_point_plan"), [asdict(point) for point in points], label=f"{study_key} ordered point plan")
    _mapping_equal(metadata.get("search_config"), asdict(search), label=f"{study_key} search config")
    _require(metadata.get("phase_space") == "full_sphere", f"{study_key} sweep is not full-sphere")
    _require(metadata.get("common_geometry_sha256") == geometry_hash, f"{study_key} geometry hash differs")
    _require_path_suffix(metadata.get("aggregate_csv"), Path("mot_simple") / campaign.CAMPAIGN_NAME / study_key / "aggregate.csv", label=f"{study_key} aggregate CSV")
    expected_figure = campaign.CampaignPaths(statistics_dir, figures_dir).relationship_plot(study_key)
    _require_path_suffix(metadata.get("relationship_plot"), Path("mot_simple") / campaign.CAMPAIGN_NAME / study_key / expected_figure.name, label=f"{study_key} relationship plot")
    return aggregate


def _validate_force_outputs(statistics_dir: Path, figures_dir: Path) -> tuple[int, int, float]:
    force_statistics = statistics_dir / "04_force_vs_detuning_27mW"
    metadata_payload = _read_json(force_statistics / "force_vs_detuning_metadata.json")
    _require(isinstance(metadata_payload, Mapping), "force metadata is malformed")
    metadata = metadata_payload
    _require(
        _integer(metadata.get("schema_version"), label="force schema version")
        == force_sweep.SCHEMA_VERSION,
        "force schema version differs from the active source contract",
    )
    _require(metadata.get("status") == "completed", "force sweep status is not completed")
    _require(metadata.get("all_convergence_checks_passed") is True, "force metadata reports a failed convergence check")
    _require(_integer(metadata.get("completed_point_count"), label="force completed points") == 111, "force sweep does not contain 111 completed points")
    _require(_integer(metadata.get("total_point_count"), label="force total points") == 111, "force sweep total-point count differs")
    _require(_integer(metadata.get("total_deterministic_evaluations"), label="force evaluation count") == 111, "force deterministic-evaluation count differs")
    _close(metadata.get("cooling_power_w_per_beam"), EXPECTED_REFERENCE_POWER_W, label="force cooling power", rtol=0.0)
    _close(metadata.get("beam_diameter_m"), EXPECTED_BEAM_DIAMETER_M, label="force beam diameter", rtol=0.0)
    _close(metadata.get("quadrupole_axial_gradient_g_per_cm"), EXPECTED_AXIAL_GRADIENT_G_PER_CM, label="force quadrupole gradient", rtol=0.0)
    _require(metadata.get("gravity_included_in_force") is False, "gravity was included in force diagnostics")
    _require(metadata.get("repumper_enabled") is False, "force diagnostics unexpectedly enable a repumper")
    _require(_integer(metadata.get("cooling_component_count"), label="force cooling component count") == 6, "force diagnostics do not use six cooling beams")
    _require(_integer(metadata.get("repump_component_count"), label="force repump component count") == 0, "force diagnostics include repump components")
    expected_signature = force_sweep._resume_signature(
        EXPECTED_FORCE_DETUNING_N, force_sweep.SimpleForceSweepNumerics()
    )
    _mapping_equal(metadata.get("resume_signature"), expected_signature, label="force resume signature")

    header, rows = _read_csv(force_statistics / "force_vs_detuning.csv")
    _require(header == list(force_sweep.CSV_FIELDNAMES), "force CSV schema differs from active source")
    _require(len(rows) == 111, "force CSV does not contain 111 rows")
    _require([_integer(row.get("point_index"), label="force point index") for row in rows] == list(range(111)), "force point indices are not canonical")
    _exact_float_grid([row.get("detuning_n") for row in rows], EXPECTED_FORCE_DETUNING_N, label="force detuning grid")
    maximum_residual = 0.0
    linewidths: set[float] = set()
    for index, row in enumerate(rows):
        n_value = EXPECTED_FORCE_DETUNING_N[index]
        numeric_values = []
        for field in force_sweep.CSV_FIELDNAMES:
            if field.endswith("converged") or field.endswith("interior") or field == "all_converged":
                continue
            numeric_values.append(_float(row.get(field), label=f"force[{index}].{field}"))
        _require(np.all(np.isfinite(numeric_values)), f"force row {index} contains nonfinite values")
        linewidth = _float(row.get("linewidth_hz"), label=f"force[{index}] linewidth")
        linewidths.add(linewidth)
        maximum_residual = max(maximum_residual, _close(row.get("detuning_hz"), n_value * linewidth, label=f"force[{index}] detuning conversion"))
        maximum_residual = max(maximum_residual, _close(row.get("cooling_power_w_per_beam"), EXPECTED_REFERENCE_POWER_W, label=f"force[{index}] power", rtol=0.0))
        maximum_residual = max(maximum_residual, _close(row.get("cooling_beam_diameter_m"), EXPECTED_BEAM_DIAMETER_M, label=f"force[{index}] beam diameter", rtol=0.0))
        _require(_strict_bool(row.get("all_converged"), label=f"force[{index}] all_converged"), f"force row {index} is not fully converged")
        for axis in "xyz":
            fine_slope = _float(row.get(f"restoring_slope_{axis}_n_per_m"), label=f"force[{index}] {axis} slope")
            coarse_slope = _float(row.get(f"restoring_slope_{axis}_coarse_n_per_m"), label=f"force[{index}] {axis} coarse slope")
            uncertainty = abs(fine_slope - coarse_slope)
            relative = uncertainty / max(abs(fine_slope), np.finfo(float).tiny)
            maximum_residual = max(maximum_residual, _close(row.get(f"restoring_slope_{axis}_numerical_uncertainty_n_per_m"), uncertainty, label=f"force[{index}] {axis} slope uncertainty", atol=1.0e-30))
            maximum_residual = max(maximum_residual, _close(row.get(f"restoring_slope_{axis}_relative_change"), relative, label=f"force[{index}] {axis} slope relative change"))
            _require(fine_slope < 0.0, f"force row {index} axis {axis} is not position restoring")
            _require(_strict_bool(row.get(f"restoring_slope_{axis}_converged"), label=f"force[{index}] {axis} slope convergence"), f"force row {index} axis {axis} slope is not converged")
            fine_turn = _float(row.get(f"turnaround_velocity_{axis}_m_per_s"), label=f"force[{index}] {axis} turnaround")
            coarse_turn = _float(row.get(f"turnaround_velocity_{axis}_coarse_m_per_s"), label=f"force[{index}] {axis} coarse turnaround")
            change = abs(fine_turn - coarse_turn)
            maximum_residual = max(maximum_residual, _close(row.get(f"turnaround_velocity_{axis}_absolute_change_m_per_s"), change, label=f"force[{index}] {axis} turnaround change"))
            maximum_residual = max(maximum_residual, _close(row.get(f"turnaround_velocity_{axis}_numerical_uncertainty_m_per_s"), change, label=f"force[{index}] {axis} turnaround uncertainty"))
            _require(0.0 < fine_turn < 50.0, f"force row {index} axis {axis} turnaround is not interior")
            _require(_strict_bool(row.get(f"turnaround_{axis}_interior"), label=f"force[{index}] {axis} turnaround interior"), f"force row {index} axis {axis} turnaround is not marked interior")
            _require(_strict_bool(row.get(f"turnaround_{axis}_converged"), label=f"force[{index}] {axis} turnaround convergence"), f"force row {index} axis {axis} turnaround is not converged")
    _require(len(linewidths) == 1, "force rows do not share one linewidth")
    return len(rows), len(rows) * 3, maximum_residual


def validate_campaign(
    statistics_dir: Path,
    figures_dir: Path,
) -> CampaignQAResult:
    """Validate a completed campaign without changing any file."""

    statistics_dir = statistics_dir.resolve()
    figures_dir = figures_dir.resolve()
    _require(statistics_dir.is_dir(), f"statistics directory does not exist: {statistics_dir}")
    _require(figures_dir.is_dir(), f"figures directory does not exist: {figures_dir}")
    _require(statistics_dir.parent.name == "mot_simple", "statistics root is not under mot_simple")
    _require(figures_dir.parent.name == "mot_simple", "figures root is not under mot_simple")
    _require(statistics_dir.name == figures_dir.name, "statistics and figure campaign roots have different names")

    metadata, search, geometry_hash, common_points = _validate_campaign_header(statistics_dir)
    point_groups = {
        study_key: campaign.build_relationship_points(study_key, values)
        for study_key, _, values in STUDY_SPECS
    }
    point_results: list[PointQAResult] = []
    for study_key, _, _ in STUDY_SPECS:
        points = point_groups[study_key]
        aggregate = _validate_study_metadata(
            study_key,
            points,
            statistics_dir=statistics_dir,
            figures_dir=figures_dir,
            search=search,
            geometry_hash=geometry_hash,
        )
        for point, aggregate_row in zip(points, aggregate, strict=True):
            point_results.append(
                _validate_point(
                    point,
                    search=search,
                    geometry_hash=geometry_hash,
                    common_points=common_points,
                    statistics_dir=statistics_dir,
                    figures_dir=figures_dir,
                    aggregate_row=aggregate_row,
                )
            )

    _require(len(point_results) == EXPECTED_LOADING_POINT_COUNT, "QA did not visit all 67 loading points")
    sample_count = sum(result.ray_count for result in point_results)
    audit_count = sum(result.audit_row_count for result in point_results)
    override_count = sum(result.override_count for result in point_results)
    indeterminate_zero_flux_ray_count = sum(
        result.indeterminate_zero_flux_ray_count for result in point_results
    )
    _require(sample_count == EXPECTED_TOTAL_RAY_COUNT, "capture sample ledgers do not total 41,875 rows")
    _require(audit_count == EXPECTED_TOTAL_RAY_COUNT, "endpoint-audit ledgers do not total 41,875 rows")
    _require(
        _integer(
            metadata.get("indeterminate_zero_flux_ray_count"),
            label="campaign indeterminate-zero count",
        )
        == indeterminate_zero_flux_ray_count,
        "campaign indeterminate-zero count differs from point evidence",
    )

    force_count, force_axis_count, force_residual = _validate_force_outputs(
        statistics_dir, figures_dir
    )
    temperature_outputs = tuple(
        path
        for root in (statistics_dir, figures_dir)
        for path in root.rglob("*")
        if "temperature" in path.name.lower()
    )
    _require(not temperature_outputs, f"temperature outputs are present: {temperature_outputs[:3]}")

    final_figures = tuple(figures_dir / relative for relative in FINAL_FIGURE_RELATIVE_PATHS)
    for path in final_figures:
        _png_dimensions(path)

    maximum_residual = max(
        [force_residual, *(result.maximum_numeric_residual for result in point_results)]
    )
    estimator_differences = []
    for result in point_results:
        point = point_groups[result.study_key][result.point_index]
        loading_payload = _read_json(
            statistics_dir
            / result.study_key
            / "points"
            / point.slug
            / "loading_rate_result.json"
        )
        _require(isinstance(loading_payload, Mapping), f"{point.slug} loading JSON is malformed")
        estimator_differences.append(
            abs(
                _float(loading_payload.get("loading_rate_mean_atoms_per_s"), label=f"{point.slug} mean loading")
                - _float(loading_payload.get("loading_rate_from_mean_spectrum_atoms_per_s"), label=f"{point.slug} spectrum loading")
            )
        )

    return CampaignQAResult(
        outcome="PASS",
        campaign_name=str(metadata["campaign_name"]),
        campaign_signature_sha256=str(metadata["campaign_signature_sha256"]),
        geometry_sha256=geometry_hash,
        loading_point_count=len(point_results),
        capture_sample_row_count=sample_count,
        endpoint_audit_row_count=audit_count,
        velocity_override_count=override_count,
        indeterminate_zero_flux_ray_count=indeterminate_zero_flux_ray_count,
        maximum_numeric_residual=maximum_residual,
        maximum_loading_estimator_difference_atoms_per_s=max(estimator_differences, default=0.0),
        force_point_count=force_count,
        force_axis_check_count=force_axis_count,
        temperature_output_count=len(temperature_outputs),
        final_figure_count=len(final_figures),
        final_figures=tuple(str(path) for path in final_figures),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        result = validate_campaign(args.statistics_dir, args.figures_dir)
    except CampaignQAFailure as exc:
        print(json.dumps({"outcome": "FAIL", "reason": str(exc)}, indent=2), flush=True)
        return 1
    print(json.dumps(asdict(result), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CampaignQAFailure",
    "CampaignQAResult",
    "EXPECTED_TOTAL_RAY_COUNT",
    "FINAL_FIGURE_RELATIVE_PATHS",
    "build_argument_parser",
    "main",
    "validate_campaign",
]
