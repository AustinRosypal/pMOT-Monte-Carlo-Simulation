"""Refined, independently executed relationship sweeps for the 24-state MOT.

This module records the September 2026 refinement as a new campaign rather
than changing the grids or output provenance of :mod:`relationship_sweeps`.
The three loading studies are executed serially and reuse one seeded set of
full-sphere direction discs and uniform-area disc points.  They are never
combined into a Cartesian parameter product.

Temperature is deliberately a separate physical experiment: it uses
independent preloaded Langevin clouds, not incident launch discs.  The same
seeded initial clouds are reused at every detuning so that cooling detuning is
the only changed physical parameter.  The force sweep is deterministic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configuration import multilevel_mot_paths
from .force_sweep import (
    ForceSweepNumerics,
    plot_damping_turnarounds_vs_detuning,
    plot_restoring_slopes_vs_detuning,
    run_force_detuning_sweep,
)
from .power_loading_study import REPUMP_POWER_W_PER_BEAM
from .rate_capture import RateCaptureSearchConfig
from .relationship_sweeps import (
    CampaignPaths,
    DEFAULT_COOLING_POWER_W_PER_BEAM,
    DETUNING_STUDY_KEY,
    EFFECTIVE_STUDY_KEY,
    RAW_STUDY_KEY,
    RelationshipPoint,
    build_relationship_points,
    effective_saturation_from_s0,
    on_resonance_saturation_parameter,
    run_loading_relationship,
)
from .temperature_sweep import (
    plot_temperature_vs_detuning,
    run_temperature_detuning_sweep,
)


RAW_SATURATION_VALUES: tuple[float, ...] = (
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
"""Existing 16 on-resonance saturation points plus the eight refinements."""

EFFECTIVE_SATURATION_VALUES: tuple[float, ...] = tuple(
    0.25 * index for index in range(1, 21)
)
"""Twenty effective-saturation points from 0.25 through 5.00 inclusive."""

DETUNING_N_VALUES: tuple[float, ...] = tuple(
    -0.25 * index for index in range(2, 25)
)
"""Twenty-three detunings from -0.5 through -6.0 in steps of -0.25."""

FORCE_DETUNING_N_VALUES: tuple[float, ...] = tuple(
    -0.5 - 0.05 * index for index in range(111)
)
"""Dense deterministic force grid from -0.5 through -6.0 in 0.05 steps."""

CAMPAIGN_NAME = (
    "refined_relationships_full_sphere_25x25_r15mm_"
    "27mW_reference_repump0p1mW_20260903"
)
DEFAULT_DISC_COUNT = 25
DEFAULT_POINTS_PER_DISC = 25
DEFAULT_DISC_RADIUS_M = 15.0e-3
DEFAULT_SEED = 20260903
DEFAULT_WORKER_COUNT = min(24, os.cpu_count() or 1)
TEMPERATURE_ENSEMBLE_COUNT = 25
TEMPERATURE_ATOMS_PER_ENSEMBLE = 25
TEMPERATURE_SEED = DEFAULT_SEED
CAMPAIGN_SCHEMA_VERSION = 1

RAW_STAGE = "raw_saturation_loading"
EFFECTIVE_STAGE = "effective_saturation_loading"
DETUNING_LOADING_STAGE = "detuning_loading"
TEMPERATURE_STAGE = "detuning_temperature"
FORCE_STAGE = "detuning_force"
STAGE_ORDER: tuple[str, ...] = (
    RAW_STAGE,
    EFFECTIVE_STAGE,
    DETUNING_LOADING_STAGE,
    TEMPERATURE_STAGE,
    FORCE_STAGE,
)

_STAGE_TO_STUDY_KEY = {
    RAW_STAGE: RAW_STUDY_KEY,
    EFFECTIVE_STAGE: EFFECTIVE_STUDY_KEY,
    DETUNING_LOADING_STAGE: DETUNING_STUDY_KEY,
}


def default_refined_campaign_paths(root: Path | None = None) -> CampaignPaths:
    """Return isolated statistics and figure roots for this refinement."""

    paths = multilevel_mot_paths(root)
    return CampaignPaths(
        statistics=paths["statistics"] / CAMPAIGN_NAME,
        figures=paths["figures"] / CAMPAIGN_NAME,
    )


def default_refined_search_config(
    *, worker_count: int = DEFAULT_WORKER_COUNT, seed: int = DEFAULT_SEED
) -> RateCaptureSearchConfig:
    """Return the requested 25 x 25, r=15 mm full-sphere launch design."""

    if worker_count <= 0 or worker_count > 24:
        raise ValueError("worker_count must be in the range 1..24")
    return replace(
        RateCaptureSearchConfig(),
        disc_radius_m=DEFAULT_DISC_RADIUS_M,
        disc_count=DEFAULT_DISC_COUNT,
        points_per_disc=DEFAULT_POINTS_PER_DISC,
        include_center_point=False,
        seed=int(seed),
        worker_count=int(worker_count),
        phase_space="full_sphere",
    )


def requested_refined_points() -> dict[str, tuple[RelationshipPoint, ...]]:
    """Return the exact, independent point plans for the loading studies."""

    return {
        RAW_STUDY_KEY: build_relationship_points(
            RAW_STUDY_KEY, RAW_SATURATION_VALUES
        ),
        EFFECTIVE_STUDY_KEY: build_relationship_points(
            EFFECTIVE_STUDY_KEY, EFFECTIVE_SATURATION_VALUES
        ),
        DETUNING_STUDY_KEY: build_relationship_points(
            DETUNING_STUDY_KEY, DETUNING_N_VALUES
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    temporary.replace(path)


def _campaign_signature(search_config: RateCaptureSearchConfig) -> str:
    search = asdict(search_config)
    # Worker count changes throughput, not the numerical campaign definition.
    search.pop("worker_count", None)
    point_groups = requested_refined_points()
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "model": "24-state repumper-included multilevel population-rate MOT",
        "search_config": search,
        "point_groups": {
            key: [asdict(point) for point in points]
            for key, points in point_groups.items()
        },
        "temperature": {
            "detuning_n_values": list(DETUNING_N_VALUES),
            "ensemble_realization_count": TEMPERATURE_ENSEMBLE_COUNT,
            "atoms_per_ensemble": TEMPERATURE_ATOMS_PER_ENSEMBLE,
            "seed": TEMPERATURE_SEED,
            "cooling_power_w_per_beam": DEFAULT_COOLING_POWER_W_PER_BEAM,
        },
        "force": {
            "detuning_n_values": list(FORCE_DETUNING_N_VALUES),
            "numerics": asdict(ForceSweepNumerics()),
            "deterministic_evaluations_per_detuning": 1,
            "cooling_power_w_per_beam": DEFAULT_COOLING_POWER_W_PER_BEAM,
        },
        "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _new_campaign_metadata(
    paths: CampaignPaths,
    search_config: RateCaptureSearchConfig,
) -> dict[str, object]:
    point_groups = requested_refined_points()
    loading_point_count = sum(len(points) for points in point_groups.values())
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_name": CAMPAIGN_NAME,
        "campaign_signature_sha256": _campaign_signature(search_config),
        "status": "pending",
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "model": "24-state repumper-included multilevel population-rate MOT",
        "execution_order": list(STAGE_ORDER),
        "combinatorial_product_used": False,
        "loading_studies_are_independent_and_sequential": True,
        "search_config": asdict(search_config),
        "loading_point_count": loading_point_count,
        "capture_threshold_search_count": (
            loading_point_count
            * search_config.disc_count
            * search_config.points_per_disc
        ),
        "shared_seeded_loading_geometry": {
            "seed": search_config.seed,
            "description": (
                "The same full-sphere direction discs and uniform-area disc "
                "coordinates are reused at every loading point."
            ),
        },
        "ordered_grids": {
            "s0": list(RAW_SATURATION_VALUES),
            "s_eff": list(EFFECTIVE_SATURATION_VALUES),
            "detuning_loading_delta_over_gamma": list(DETUNING_N_VALUES),
            "detuning_temperature_delta_over_gamma": list(DETUNING_N_VALUES),
            "detuning_force_delta_over_gamma": list(FORCE_DETUNING_N_VALUES),
        },
        "temperature_design": {
            "independent_preloaded_cloud_count": TEMPERATURE_ENSEMBLE_COUNT,
            "atoms_per_cloud": TEMPERATURE_ATOMS_PER_ENSEMBLE,
            "trajectories_per_detuning": (
                TEMPERATURE_ENSEMBLE_COUNT * TEMPERATURE_ATOMS_PER_ENSEMBLE
            ),
            "seed": TEMPERATURE_SEED,
            "physical_interpretation": (
                "Temperature is computed from independent preloaded Langevin "
                "clouds rather than incident sampling discs; common initial "
                "clouds are reused and only cooling detuning changes."
            ),
            "theory_overlay": (
                "detuning-dependent multilevel Doppler reference implemented "
                "by temperature_sweep.plot_temperature_vs_detuning"
            ),
        },
        "fixed_parameters": {
            "detuning_and_temperature_cooling_power_w_per_beam": (
                DEFAULT_COOLING_POWER_W_PER_BEAM
            ),
            "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
            "disc_radius_m": DEFAULT_DISC_RADIUS_M,
        },
        "saturation_power_note": (
            "The raw-s0 and effective-saturation studies vary cooling power to "
            "realize the requested saturation value at fixed -15 MHz detuning. "
            "The 27 mW cooling power is the reference marker on those plots and "
            "is fixed only for the detuning-loading, temperature, and force studies."
        ),
        "stage_status": {stage: "pending" for stage in STAGE_ORDER},
        "results": {},
        "statistics_root": str(paths.statistics.resolve()),
        "figures_root": str(paths.figures.resolve()),
    }


def _load_or_initialize_metadata(
    paths: CampaignPaths,
    search_config: RateCaptureSearchConfig,
    *,
    resume: bool,
) -> dict[str, object]:
    paths.statistics.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    expected_signature = _campaign_signature(search_config)
    if resume and paths.metadata_json.is_file():
        metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        if metadata.get("campaign_signature_sha256") != expected_signature:
            raise ValueError("refined campaign resume signature mismatch")
        metadata["search_config"] = asdict(search_config)
        metadata["updated_utc"] = _utc_now()
        return metadata
    metadata = _new_campaign_metadata(paths, search_config)
    _atomic_write_json(paths.metadata_json, metadata)
    return metadata


def _force_statistics(paths: CampaignPaths) -> Path:
    return paths.statistics / "04_force_vs_detuning_27mW"


def _force_figures(paths: CampaignPaths) -> Path:
    return paths.figures / "04_force_vs_detuning_27mW"


def plot_refined_loading_relationship(
    rows: Sequence[Mapping[str, object]],
    study_key: str,
    path: Path,
    *,
    search_config: RateCaptureSearchConfig,
) -> Path:
    """Plot a refined loading sweep with its requested horizontal limits."""

    if not rows:
        raise ValueError("at least one aggregate row is required")
    labels = {
        RAW_STUDY_KEY: (
            r"On-resonance saturation parameter $s_0$",
            r"Loading Rate vs. On-Resonance Saturation ($\Delta/2\pi=-15$ MHz)",
            "s0",
            (0.0, 125.0),
        ),
        EFFECTIVE_STUDY_KEY: (
            r"Effective saturation parameter $s_{\mathrm{eff}}$",
            r"Loading Rate vs. Effective Saturation ($\Delta/2\pi=-15$ MHz)",
            "seff",
            (0.0, 5.0),
        ),
        DETUNING_STUDY_KEY: (
            r"Cooling detuning $\Delta/\Gamma$",
            "Loading Rate vs. Cooling Detuning (27 mW per cooling beam)",
            "detuning_n",
            (-6.0, -0.5),
        ),
    }
    if study_key not in labels:
        raise ValueError(f"unknown refined relationship study: {study_key}")
    xlabel, title, xfield, limits = labels[study_key]
    ordered = sorted(rows, key=lambda row: float(row[xfield]))
    x = np.asarray([float(row[xfield]) for row in ordered], dtype=float)
    mean = np.asarray(
        [float(row["loading_rate_mean_atoms_per_s"]) for row in ordered],
        dtype=float,
    )
    lower = np.asarray(
        [float(row["loading_rate_t95_lower_atoms_per_s"]) for row in ordered],
        dtype=float,
    )
    upper = np.asarray(
        [float(row["loading_rate_t95_upper_atoms_per_s"]) for row in ordered],
        dtype=float,
    )
    scale = 1.0e6
    figure, axis = plt.subplots(figsize=(9.4, 6.8))
    figure.subplots_adjust(left=0.125, right=0.985, top=0.91, bottom=0.22)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.errorbar(
        x,
        mean / scale,
        yerr=np.vstack((mean - lower, upper - mean)) / scale,
        fmt="o-",
        color="#0f766e",
        ecolor="#9f4a13",
        linewidth=1.8,
        markersize=5.5,
        capsize=3,
        label="Mean ± 95% CI",
    )
    if study_key == RAW_STUDY_KEY:
        # Retain the requested 0--125 overview while making the closely spaced
        # 0.25--5 points independently readable.  The inset repeats data; it
        # does not introduce interpolation or another calculation.
        low_s_mask = x <= 5.0
        inset = axis.inset_axes((0.48, 0.12, 0.28, 0.30))
        inset.errorbar(
            x[low_s_mask],
            mean[low_s_mask] / scale,
            yerr=np.vstack(
                (
                    mean[low_s_mask] - lower[low_s_mask],
                    upper[low_s_mask] - mean[low_s_mask],
                )
            )
            / scale,
            fmt="o-",
            color="#0f766e",
            ecolor="#9f4a13",
            linewidth=1.15,
            markersize=3.5,
            capsize=2,
        )
        inset.set(
            title=r"low-$s_0$ detail",
            xlim=(0.0, 5.15),
            ylim=(0.0, None),
        )
        inset.set_xticks((0.0, 1.0, 2.0, 3.0, 5.0))
        inset.tick_params(axis="both", labelsize=7)
        inset.title.set_fontsize(8.5)
        inset.grid(True, alpha=0.20, linewidth=0.6)
    if study_key in {RAW_STUDY_KEY, EFFECTIVE_STUDY_KEY}:
        reference_s0 = on_resonance_saturation_parameter(
            DEFAULT_COOLING_POWER_W_PER_BEAM
        )
        if study_key == RAW_STUDY_KEY:
            reference_x = reference_s0
        else:
            reference_n = float(ordered[0]["cooling_detuning_n"])
            reference_x = effective_saturation_from_s0(reference_s0, reference_n)
        axis.axvline(
            reference_x,
            color="#475569",
            linestyle="--",
            linewidth=1.2,
            label="27 mW reference",
        )
    horizontal_span = limits[1] - limits[0]
    if study_key == DETUNING_STUDY_KEY:
        plotted_limits = (
            limits[0] - 0.02 * horizontal_span,
            limits[1] + 0.02 * horizontal_span,
        )
    else:
        plotted_limits = (limits[0], limits[1] + 0.02 * horizontal_span)
    axis.set(
        title=title,
        xlabel=xlabel,
        ylabel=r"Loading rate [$10^6$ atoms/s]",
        xlim=plotted_limits,
    )
    if study_key == RAW_STUDY_KEY:
        axis.set_xticks((0.0, 25.0, 50.0, 75.0, 100.0, 125.0))
    else:
        requested_ticks = np.asarray(axis.get_xticks(), dtype=float)
        requested_ticks = requested_ticks[
            (requested_ticks >= limits[0]) & (requested_ticks <= limits[1])
        ]
        axis.set_xticks(np.unique(np.append(requested_ticks, limits)))
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    axis.text(
        0.0,
        -0.18,
        (
            f"{search_config.disc_count} full-sphere direction discs x "
            f"{search_config.points_per_disc} random points; r = "
            f"{1.0e3 * search_config.disc_radius_m:g} mm\n"
            "Intervals cluster by direction disc "
            f"(df = {search_config.disc_count - 1})"
        ),
        transform=axis.transAxes,
        fontsize=8.5,
        color="#334155",
        va="top",
        clip_on=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return path


def run_refined_loading_study(
    study_key: str,
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    search_config: RateCaptureSearchConfig | None = None,
    paths: CampaignPaths | None = None,
    resume: bool = True,
    plot_only: bool = False,
    overall_offset: int = 0,
    overall_total: int | None = None,
) -> list[dict[str, object]]:
    """Execute one complete loading relationship with no parameter product."""

    groups = requested_refined_points()
    if study_key not in groups:
        raise ValueError(f"unknown refined loading study: {study_key}")
    search = search_config or default_refined_search_config(
        worker_count=worker_count
    )
    campaign_paths = paths or default_refined_campaign_paths()
    points = groups[study_key]
    rows = run_loading_relationship(
        study_key,
        points,
        search_config=search,
        paths=campaign_paths,
        worker_count=worker_count,
        resume=resume,
        plot_only=plot_only,
        overall_offset=overall_offset,
        overall_total=len(points) if overall_total is None else overall_total,
    )
    # The shared historical plotter intentionally has automatic limits.  This
    # refinement requests explicit comparable ranges, so overwrite only this
    # campaign's final plot with its power/range-qualified presentation.
    plot_refined_loading_relationship(
        rows,
        study_key,
        campaign_paths.relationship_plot(study_key),
        search_config=search,
    )
    return rows


def run_refined_temperature_sweep(
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    paths: CampaignPaths | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """Run 25 independent preloaded clouds x 25 atoms at each detuning."""

    if worker_count <= 0 or worker_count > 24:
        raise ValueError("worker_count must be in the range 1..24")
    campaign_paths = paths or default_refined_campaign_paths()
    result = run_temperature_detuning_sweep(
        ensemble_realization_count=TEMPERATURE_ENSEMBLE_COUNT,
        atoms_per_ensemble=TEMPERATURE_ATOMS_PER_ENSEMBLE,
        worker_count=worker_count,
        seed=TEMPERATURE_SEED,
        output_directory=campaign_paths.temperature_statistics,
        figure_directory=campaign_paths.temperature_figures,
        resume=resume,
        detuning_n_values=DETUNING_N_VALUES,
        cooling_power_w_per_beam=DEFAULT_COOLING_POWER_W_PER_BEAM,
    )
    if result.get("status") != "completed" or int(
        result.get("completed_point_count", -1)
    ) != len(DETUNING_N_VALUES):
        raise RuntimeError("refined temperature sweep did not complete every detuning")
    return result


def run_refined_force_sweep(
    *,
    paths: CampaignPaths | None = None,
    resume: bool = True,
    numerics: ForceSweepNumerics | None = None,
) -> dict[str, object]:
    """Run the dense deterministic restoring/damping detuning sweep."""

    campaign_paths = paths or default_refined_campaign_paths()
    result = run_force_detuning_sweep(
        detuning_n_values=FORCE_DETUNING_N_VALUES,
        numerics=numerics or ForceSweepNumerics(),
        output_directory=_force_statistics(campaign_paths),
        figure_directory=_force_figures(campaign_paths),
        resume=resume,
    )
    if result.get("status") != "completed" or int(
        result.get("completed_point_count", -1)
    ) != len(FORCE_DETUNING_N_VALUES):
        raise RuntimeError("refined deterministic force sweep is incomplete")
    return result


def _loading_result_summary(
    study_key: str,
    rows: Sequence[Mapping[str, object]],
    paths: CampaignPaths,
) -> dict[str, object]:
    return {
        "status": "completed",
        "completed_point_count": len(rows),
        "aggregate_csv": str(paths.aggregate_csv(study_key).resolve()),
        "plot": str(paths.relationship_plot(study_key).resolve()),
    }


def _update_stage(
    metadata: dict[str, object],
    paths: CampaignPaths,
    stage: str,
    status: str,
    *,
    result: Mapping[str, object] | None = None,
    error: BaseException | None = None,
) -> None:
    stage_status = dict(metadata.get("stage_status", {}))
    stage_status[stage] = status
    metadata["stage_status"] = stage_status
    metadata["current_stage"] = stage
    metadata["updated_utc"] = _utc_now()
    if result is not None:
        results = dict(metadata.get("results", {}))
        results[stage] = dict(result)
        metadata["results"] = results
    if error is not None:
        metadata["last_error"] = f"{type(error).__name__}: {error}"
    if all(stage_status.get(item) == "completed" for item in STAGE_ORDER):
        metadata["status"] = "completed"
    elif status == "completed":
        metadata["status"] = "running"
    else:
        metadata["status"] = status
    _atomic_write_json(paths.metadata_json, metadata)


def run_refined_campaign(
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    paths: CampaignPaths | None = None,
    resume: bool = True,
    selected_stages: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run selected stages in canonical order, or the full campaign serially."""

    if worker_count <= 0 or worker_count > 24:
        raise ValueError("worker_count must be in the range 1..24")
    campaign_paths = paths or default_refined_campaign_paths()
    search = default_refined_search_config(worker_count=worker_count)
    metadata = _load_or_initialize_metadata(
        campaign_paths, search, resume=resume
    )
    requested = set(STAGE_ORDER if selected_stages is None else selected_stages)
    unknown = requested.difference(STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown refined campaign stages: {sorted(unknown)}")
    ordered = tuple(stage for stage in STAGE_ORDER if stage in requested)
    if not ordered:
        raise ValueError("at least one refined campaign stage is required")

    loading_groups = requested_refined_points()
    loading_total = sum(len(points) for points in loading_groups.values())
    loading_offsets = {
        RAW_STAGE: 0,
        EFFECTIVE_STAGE: len(loading_groups[RAW_STUDY_KEY]),
        DETUNING_LOADING_STAGE: (
            len(loading_groups[RAW_STUDY_KEY])
            + len(loading_groups[EFFECTIVE_STUDY_KEY])
        ),
    }
    for stage in ordered:
        _update_stage(metadata, campaign_paths, stage, "running")
        print(f"[refined-campaign] starting stage: {stage}", flush=True)
        try:
            if stage in _STAGE_TO_STUDY_KEY:
                study_key = _STAGE_TO_STUDY_KEY[stage]
                rows = run_refined_loading_study(
                    study_key,
                    worker_count=worker_count,
                    search_config=search,
                    paths=campaign_paths,
                    resume=resume,
                    overall_offset=loading_offsets[stage],
                    overall_total=loading_total,
                )
                summary: Mapping[str, object] = _loading_result_summary(
                    study_key, rows, campaign_paths
                )
            elif stage == TEMPERATURE_STAGE:
                result = run_refined_temperature_sweep(
                    worker_count=worker_count,
                    paths=campaign_paths,
                    resume=resume,
                )
                summary = {
                    "status": result["status"],
                    "completed_point_count": result["completed_point_count"],
                    "outputs": result["outputs"],
                }
            else:
                result = run_refined_force_sweep(
                    paths=campaign_paths,
                    resume=resume,
                )
                summary = {
                    "status": result["status"],
                    "completed_point_count": result["completed_point_count"],
                    "outputs": result["outputs"],
                }
        except BaseException as exc:
            failure_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _update_stage(
                metadata,
                campaign_paths,
                stage,
                failure_status,
                error=exc,
            )
            raise
        _update_stage(
            metadata,
            campaign_paths,
            stage,
            "completed",
            result=summary,
        )
        print(f"[refined-campaign] completed stage: {stage}", flush=True)

    metadata = json.loads(campaign_paths.metadata_json.read_text(encoding="utf-8"))
    if not all(
        metadata["stage_status"].get(stage) == "completed" for stage in STAGE_ORDER
    ):
        metadata["status"] = "partially_completed"
        metadata["updated_utc"] = _utc_now()
        _atomic_write_json(campaign_paths.metadata_json, metadata)
    return metadata


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def regenerate_refined_plots(
    *, paths: CampaignPaths | None = None
) -> dict[str, str]:
    """Regenerate every final campaign plot from completed checkpoint data."""

    campaign_paths = paths or default_refined_campaign_paths()
    search = default_refined_search_config()
    outputs: dict[str, str] = {}
    for study_key in (RAW_STUDY_KEY, EFFECTIVE_STUDY_KEY, DETUNING_STUDY_KEY):
        plot = plot_refined_loading_relationship(
            _read_csv_rows(campaign_paths.aggregate_csv(study_key)),
            study_key,
            campaign_paths.relationship_plot(study_key),
            search_config=search,
        )
        outputs[f"{study_key}_plot"] = str(plot.resolve())

    temperature_csv = (
        campaign_paths.temperature_statistics / "temperature_vs_detuning.csv"
    )
    combined_temperature_plot = (
        campaign_paths.temperature_figures / "temperature_vs_detuning.png"
    )
    temperature_only_plot = (
        campaign_paths.temperature_figures / "temperature_only_vs_detuning.png"
    )
    plot_temperature_vs_detuning(
        temperature_csv,
        combined_temperature_plot,
        cooling_power_w_per_beam=DEFAULT_COOLING_POWER_W_PER_BEAM,
        ensemble_realization_count=TEMPERATURE_ENSEMBLE_COUNT,
        atoms_per_ensemble=TEMPERATURE_ATOMS_PER_ENSEMBLE,
        include_survivor_panel=True,
    )
    plot_temperature_vs_detuning(
        temperature_csv,
        temperature_only_plot,
        cooling_power_w_per_beam=DEFAULT_COOLING_POWER_W_PER_BEAM,
        ensemble_realization_count=TEMPERATURE_ENSEMBLE_COUNT,
        atoms_per_ensemble=TEMPERATURE_ATOMS_PER_ENSEMBLE,
        include_survivor_panel=False,
    )
    outputs["temperature_plot"] = str(combined_temperature_plot.resolve())
    outputs["temperature_only_plot"] = str(temperature_only_plot.resolve())

    force_csv = _force_statistics(campaign_paths) / "force_vs_detuning.csv"
    restoring_plot = _force_figures(campaign_paths) / "restoring_slope_vs_detuning.png"
    turnaround_plot = (
        _force_figures(campaign_paths) / "damping_turnaround_vs_detuning.png"
    )
    plot_restoring_slopes_vs_detuning(force_csv, restoring_plot)
    plot_damping_turnarounds_vs_detuning(force_csv, turnaround_plot)
    outputs["restoring_slope_plot"] = str(restoring_plot.resolve())
    outputs["damping_turnaround_plot"] = str(turnaround_plot.resolve())
    return outputs


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated refined 24-state MOT relationship campaign"
    )
    parser.add_argument(
        "study",
        choices=(
            "campaign",
            "raw-s",
            "effective-s",
            "detuning-loading",
            "temperature",
            "force",
            "plots",
        ),
        nargs="?",
        default="campaign",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    paths = default_refined_campaign_paths(args.project_root)
    if args.study == "plots":
        result: Mapping[str, object] = regenerate_refined_plots(paths=paths)
    else:
        selection = {
            "raw-s": (RAW_STAGE,),
            "effective-s": (EFFECTIVE_STAGE,),
            "detuning-loading": (DETUNING_LOADING_STAGE,),
            "temperature": (TEMPERATURE_STAGE,),
            "force": (FORCE_STAGE,),
        }
        result = run_refined_campaign(
            worker_count=args.workers,
            paths=paths,
            resume=args.resume,
            selected_stages=None if args.study == "campaign" else selection[args.study],
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_NAME",
    "DEFAULT_DISC_COUNT",
    "DEFAULT_DISC_RADIUS_M",
    "DEFAULT_POINTS_PER_DISC",
    "DEFAULT_SEED",
    "DEFAULT_WORKER_COUNT",
    "DETUNING_LOADING_STAGE",
    "DETUNING_N_VALUES",
    "EFFECTIVE_STAGE",
    "EFFECTIVE_SATURATION_VALUES",
    "FORCE_STAGE",
    "FORCE_DETUNING_N_VALUES",
    "RAW_STAGE",
    "RAW_SATURATION_VALUES",
    "STAGE_ORDER",
    "TEMPERATURE_ATOMS_PER_ENSEMBLE",
    "TEMPERATURE_ENSEMBLE_COUNT",
    "TEMPERATURE_SEED",
    "TEMPERATURE_STAGE",
    "build_argument_parser",
    "default_refined_campaign_paths",
    "default_refined_search_config",
    "main",
    "plot_refined_loading_relationship",
    "regenerate_refined_plots",
    "requested_refined_points",
    "run_refined_campaign",
    "run_refined_force_sweep",
    "run_refined_loading_study",
    "run_refined_temperature_sweep",
]
