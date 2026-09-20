"""Independent full-sphere loading-relationship campaigns for the 24-state MOT.

The three requested studies are deliberately sequential, not Cartesian:

1. loading rate versus the on-resonance single-beam saturation ``s0``;
2. loading rate versus the detuning-reduced single-beam saturation ``s_eff``;
3. loading rate versus signed cooling detuning ``Delta/Gamma``.

The completed saturation studies retain their seeded 30-disc by 30-point
full-sphere products.  The remaining detuning loading study uses a separate
seeded 15-disc by 15-point full-sphere output root.  Direction discs, rather
than individual launch points, are the independent clusters used for
Student-t uncertainty intervals.  The associated temperature study likewise
uses 15 independent preloaded clouds with 15 atoms per cloud.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import pi
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .configuration import default_multilevel_mot_config, multilevel_mot_paths
from .power_loading_study import (
    COOLING_DETUNING_HZ,
    REPUMP_POWER_W_PER_BEAM,
    StudyPaths,
    generate_study_geometry,
    geometry_rows,
    geometry_sha256,
    run_power_loading_study,
)
from .rate_capture import RateCaptureSearchConfig
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
)

EFFECTIVE_SATURATION_VALUES: tuple[float, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0,
    12.0,
    15.0,
    18.0,
    20.0,
    22.0,
    25.0,
)

DETUNING_N_VALUES: tuple[float, ...] = (
    -0.1,
    -0.25,
    -0.5,
    -0.75,
    -1.0,
    -2.0,
    -2.5,
    -3.0,
    -4.0,
    -5.0,
    -7.0,
    -10.0,
    -12.0,
    -15.0,
)

SATURATION_CAMPAIGN_NAME = (
    "loading_relationship_sweeps_full_sphere_30x30_r15mm_repump0p1mW"
)
REMAINING_CAMPAIGN_NAME = (
    "detuning_loading_temperature_full_sphere_15x15_r15mm_27mW_repump0p1mW"
)
# Backwards-compatible name for the already-created saturation output root.
CAMPAIGN_NAME = SATURATION_CAMPAIGN_NAME
RAW_STUDY_KEY = "01_raw_saturation"
EFFECTIVE_STUDY_KEY = "02_effective_saturation"
DETUNING_STUDY_KEY = "03_detuning"
DEFAULT_COOLING_POWER_W_PER_BEAM = 27.0e-3
DEFAULT_DISC_RADIUS_M = 15.0e-3
SATURATION_DISC_COUNT = 30
SATURATION_POINTS_PER_DISC = 30
DEFAULT_DISC_COUNT = 15
DEFAULT_POINTS_PER_DISC = 15
DEFAULT_SEED = 20260830
DEFAULT_WORKER_COUNT = min(24, os.cpu_count() or 1)
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_CHECKPOINT_EVERY = 50
CAMPAIGN_SCHEMA_VERSION = 1
REFERENCE_SECONDS_PER_LOADING_POINT = 14.0 * 60.0
TEMPERATURE_ENSEMBLE_COUNT = 15
TEMPERATURE_ATOMS_PER_ENSEMBLE = 15
TEMPERATURE_SEED = 20260830


AGGREGATE_FIELDNAMES: tuple[str, ...] = (
    "point_index",
    "study_key",
    "scan_variable",
    "scan_value",
    "s0",
    "seff",
    "detuning_n",
    "cooling_power_w_per_beam",
    "cooling_power_mw_per_beam",
    "cooling_beam_diameter_m",
    "cooling_beam_diameter_mm",
    "cooling_beam_center_peak_intensity_w_per_m2",
    "cooling_beam_center_on_resonance_saturation_parameter",
    "cooling_beam_center_effective_saturation_parameter",
    "cooling_detuning_n",
    "cooling_detuning_hz",
    "cooling_detuning_mhz",
    "cooling_detuning_rad_per_s",
    "natural_linewidth_hz",
    "natural_linewidth_rad_per_s",
    "repump_power_w_per_beam",
    "repump_power_mw_per_beam",
    "loading_rate_mean_atoms_per_s",
    "loading_rate_from_mean_spectrum_atoms_per_s",
    "loading_rate_sample_std_atoms_per_s",
    "loading_rate_disc_cluster_sem_atoms_per_s",
    "loading_rate_t95_lower_atoms_per_s",
    "loading_rate_t95_upper_atoms_per_s",
    "student_t_critical_95",
    "confidence_level",
    "disc_count",
    "points_per_disc",
    "capture_threshold_search_count",
    "disc_radius_m",
    "disc_radius_mm",
    "phase_space",
    "geometry_sha256",
    "run_signature_sha256",
    "point_elapsed_wall_time_s",
    "statistics_directory",
    "figures_directory",
    "capture_cross_section_csv",
    "capture_cross_section_plot",
    "status",
)


@dataclass(frozen=True, slots=True)
class RelationshipPoint:
    """One independent parameter point in one of the three sweeps."""

    study_key: str
    point_index: int
    scan_variable: str
    scan_value: float
    cooling_power_w_per_beam: float
    cooling_detuning_n: float
    cooling_detuning_hz: float
    on_resonance_saturation: float
    effective_saturation: float

    @property
    def slug(self) -> str:
        value = _number_slug(self.scan_value)
        return f"{self.point_index:03d}_{self.scan_variable}_{value}"


@dataclass(frozen=True, slots=True)
class CampaignPaths:
    """Stable resumable output roots for the relationship campaign."""

    statistics: Path
    figures: Path

    @property
    def metadata_json(self) -> Path:
        return self.statistics / "campaign_metadata.json"

    def study_statistics(self, study_key: str) -> Path:
        return self.statistics / study_key

    def study_figures(self, study_key: str) -> Path:
        return self.figures / study_key

    def aggregate_csv(self, study_key: str) -> Path:
        return self.study_statistics(study_key) / "aggregate.csv"

    def study_metadata_json(self, study_key: str) -> Path:
        return self.study_statistics(study_key) / "sweep_metadata.json"

    def relationship_plot(self, study_key: str) -> Path:
        names = {
            RAW_STUDY_KEY: "loading_rate_vs_saturation_parameter.png",
            EFFECTIVE_STUDY_KEY: "loading_rate_vs_effective_saturation_parameter.png",
            DETUNING_STUDY_KEY: "loading_rate_vs_detuning.png",
        }
        return self.study_figures(study_key) / names[study_key]

    def point_paths(self, point: RelationshipPoint) -> StudyPaths:
        return StudyPaths(
            statistics=self.study_statistics(point.study_key) / "points" / point.slug,
            figures=self.study_figures(point.study_key) / "points" / point.slug,
        )

    @property
    def temperature_statistics(self) -> Path:
        return self.study_statistics(DETUNING_STUDY_KEY) / "temperature"

    @property
    def temperature_figures(self) -> Path:
        return self.study_figures(DETUNING_STUDY_KEY) / "temperature"


def default_campaign_paths(root: Path | None = None) -> CampaignPaths:
    """Return the preserved 30x30 saturation-campaign output roots."""

    paths = multilevel_mot_paths(root)
    return CampaignPaths(
        statistics=paths["statistics"] / SATURATION_CAMPAIGN_NAME,
        figures=paths["figures"] / SATURATION_CAMPAIGN_NAME,
    )


def default_remaining_campaign_paths(root: Path | None = None) -> CampaignPaths:
    """Return the separate 15x15 detuning-loading/temperature output roots."""

    paths = multilevel_mot_paths(root)
    return CampaignPaths(
        statistics=paths["statistics"] / REMAINING_CAMPAIGN_NAME,
        figures=paths["figures"] / REMAINING_CAMPAIGN_NAME,
    )


def default_relationship_search_config(
    *, worker_count: int = DEFAULT_WORKER_COUNT, seed: int = DEFAULT_SEED
) -> RateCaptureSearchConfig:
    """Return the 15x15 full-sphere launch design for the remaining study."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
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


def default_saturation_search_config(
    *, worker_count: int = DEFAULT_WORKER_COUNT, seed: int = DEFAULT_SEED
) -> RateCaptureSearchConfig:
    """Return the preserved 30x30 launch design used by saturation products."""

    return replace(
        default_relationship_search_config(worker_count=worker_count, seed=seed),
        disc_count=SATURATION_DISC_COUNT,
        points_per_disc=SATURATION_POINTS_PER_DISC,
    )


def saturation_power_w_per_beam(
    saturation_parameter: float,
    *,
    beam_diameter_m: float = 12.7e-3,
    saturation_intensity_w_per_m2: float | None = None,
) -> float:
    """Convert on-resonance single-beam ``s0`` to Gaussian-beam power."""

    if saturation_intensity_w_per_m2 is None:
        saturation_intensity_w_per_m2 = (
            default_multilevel_mot_config().saturation_intensity_w_per_m2
        )
    if saturation_parameter <= 0.0 or beam_diameter_m <= 0.0:
        raise ValueError("saturation_parameter and beam_diameter_m must be positive")
    radius = 0.5 * float(beam_diameter_m)
    return float(
        saturation_parameter
        * float(saturation_intensity_w_per_m2)
        * pi
        * radius**2
        / 2.0
    )


def on_resonance_saturation_parameter(
    power_w_per_beam: float,
    *,
    beam_diameter_m: float = 12.7e-3,
    saturation_intensity_w_per_m2: float | None = None,
) -> float:
    """Return ``s0=I0/I_sat`` for one Gaussian cooling beam at its center."""

    if saturation_intensity_w_per_m2 is None:
        saturation_intensity_w_per_m2 = (
            default_multilevel_mot_config().saturation_intensity_w_per_m2
        )
    if power_w_per_beam <= 0.0 or beam_diameter_m <= 0.0:
        raise ValueError("power_w_per_beam and beam_diameter_m must be positive")
    radius = 0.5 * float(beam_diameter_m)
    peak_intensity = 2.0 * float(power_w_per_beam) / (pi * radius**2)
    return float(peak_intensity / float(saturation_intensity_w_per_m2))


def detuning_reduction_denominator(detuning_n: float) -> float:
    """Return ``1+(2 Delta/Gamma)^2`` for ``detuning_n=Delta/Gamma``."""

    if not np.isfinite(detuning_n):
        raise ValueError("detuning_n must be finite")
    return float(1.0 + (2.0 * float(detuning_n)) ** 2)


def effective_saturation_from_s0(s0: float, detuning_n: float) -> float:
    if s0 <= 0.0:
        raise ValueError("s0 must be positive")
    return float(s0 / detuning_reduction_denominator(detuning_n))


def _linewidth_hz() -> float:
    return float(default_multilevel_mot_config().natural_linewidth_rad_per_s / (2.0 * pi))


def build_relationship_points(
    study_key: str,
    values: Sequence[float],
) -> tuple[RelationshipPoint, ...]:
    """Map an exact requested grid into powers, detunings, and derived saturation."""

    config = default_multilevel_mot_config()
    gamma_hz = config.natural_linewidth_rad_per_s / (2.0 * pi)
    baseline_n = COOLING_DETUNING_HZ / gamma_hz
    reference_s0 = on_resonance_saturation_parameter(DEFAULT_COOLING_POWER_W_PER_BEAM)
    points: list[RelationshipPoint] = []
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if study_key == RAW_STUDY_KEY:
            if value <= 0.0:
                raise ValueError("raw saturation values must be positive")
            s0 = value
            detuning_n = baseline_n
            power = saturation_power_w_per_beam(s0)
            seff = effective_saturation_from_s0(s0, detuning_n)
            variable = "s0"
        elif study_key == EFFECTIVE_STUDY_KEY:
            if value <= 0.0:
                raise ValueError("effective saturation values must be positive")
            detuning_n = baseline_n
            seff = value
            s0 = seff * detuning_reduction_denominator(detuning_n)
            power = saturation_power_w_per_beam(s0)
            variable = "seff"
        elif study_key == DETUNING_STUDY_KEY:
            if value >= 0.0 or not np.isfinite(value):
                raise ValueError("detuning values must be finite and negative")
            detuning_n = value
            power = DEFAULT_COOLING_POWER_W_PER_BEAM
            s0 = reference_s0
            seff = effective_saturation_from_s0(s0, detuning_n)
            variable = "detuning_n"
        else:
            raise ValueError(f"unknown relationship study: {study_key}")
        points.append(
            RelationshipPoint(
                study_key=study_key,
                point_index=index,
                scan_variable=variable,
                scan_value=value,
                cooling_power_w_per_beam=power,
                cooling_detuning_n=detuning_n,
                cooling_detuning_hz=detuning_n * gamma_hz,
                on_resonance_saturation=s0,
                effective_saturation=seff,
            )
        )
    return tuple(points)


def requested_relationship_points() -> dict[str, tuple[RelationshipPoint, ...]]:
    return {
        RAW_STUDY_KEY: build_relationship_points(RAW_STUDY_KEY, RAW_SATURATION_VALUES),
        EFFECTIVE_STUDY_KEY: build_relationship_points(
            EFFECTIVE_STUDY_KEY, EFFECTIVE_SATURATION_VALUES
        ),
        DETUNING_STUDY_KEY: build_relationship_points(DETUNING_STUDY_KEY, DETUNING_N_VALUES),
    }


def _number_slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p").replace("+", "p")


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=AGGREGATE_FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents, encoding="utf-8", newline="")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _geometry_hash(search: RateCaptureSearchConfig) -> str:
    discs, points = generate_study_geometry(search)
    return geometry_sha256(geometry_rows(discs, points))


def _campaign_signature(
    search: RateCaptureSearchConfig,
    point_groups: Mapping[str, Sequence[RelationshipPoint]],
) -> str:
    search_payload = asdict(search)
    search_payload.pop("worker_count", None)
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "model": "24-state repumper-included multilevel population-rate MOT",
        "search": search_payload,
        "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
        "point_groups": {
            key: [asdict(point) for point in points]
            for key, points in point_groups.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _study_labels(study_key: str) -> tuple[str, str, str]:
    if study_key == RAW_STUDY_KEY:
        return (
            r"On-resonance saturation parameter $s_0$",
            r"Loading Rate vs. On-Resonance Saturation ($\Delta/2\pi=-15$ MHz)",
            "s0",
        )
    if study_key == EFFECTIVE_STUDY_KEY:
        return (
            r"Effective saturation parameter $s_{\mathrm{eff}}$",
            r"Loading Rate vs. Effective Saturation ($\Delta/2\pi=-15$ MHz)",
            "seff",
        )
    if study_key == DETUNING_STUDY_KEY:
        return (
            r"Cooling detuning $\Delta/\Gamma$",
            "Loading Rate vs. Cooling Detuning (27 mW per cooling beam)",
            "detuning_n",
        )
    raise ValueError(f"unknown relationship study: {study_key}")


def plot_loading_relationship(
    rows: Sequence[Mapping[str, object]],
    study_key: str,
    path: Path,
    *,
    search_config: RateCaptureSearchConfig | None = None,
) -> Path:
    """Plot point means with asymmetric direction-cluster Student-t intervals."""

    if not rows:
        raise ValueError("at least one aggregate row is required")
    xlabel, title, xfield = _study_labels(study_key)
    ordered = sorted(rows, key=lambda row: float(row[xfield]))
    x = np.asarray([float(row[xfield]) for row in ordered], dtype=float)
    mean = np.asarray(
        [float(row["loading_rate_mean_atoms_per_s"]) for row in ordered], dtype=float
    )
    lower = np.asarray(
        [float(row["loading_rate_t95_lower_atoms_per_s"]) for row in ordered], dtype=float
    )
    upper = np.asarray(
        [float(row["loading_rate_t95_upper_atoms_per_s"]) for row in ordered], dtype=float
    )
    scale = 1.0e6
    yerr = np.vstack((mean - lower, upper - mean)) / scale
    figure, axis = plt.subplots(figsize=(9.0, 6.1), constrained_layout=True)
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.errorbar(
        x,
        mean / scale,
        yerr=yerr,
        fmt="o-",
        color="#0f766e",
        ecolor="#9f4a13",
        linewidth=1.8,
        markersize=5.5,
        capsize=3,
        label=r"Mean $\pm$ 95% Student-t interval",
    )
    if study_key in {RAW_STUDY_KEY, EFFECTIVE_STUDY_KEY}:
        reference_s0 = on_resonance_saturation_parameter(
            DEFAULT_COOLING_POWER_W_PER_BEAM
        )
        reference_x = (
            reference_s0
            if study_key == RAW_STUDY_KEY
            else effective_saturation_from_s0(
                reference_s0, COOLING_DETUNING_HZ / _linewidth_hz()
            )
        )
        axis.axvline(
            reference_x,
            color="#475569",
            linestyle="--",
            linewidth=1.2,
            label="27 mW reference",
        )
    axis.set(
        title=title,
        xlabel=xlabel,
        ylabel=r"Loading rate [$10^6$ atoms/s]",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    if search_config is not None:
        design_note = (
            f"{search_config.disc_count} full-sphere direction discs × "
            f"{search_config.points_per_disc} random points; r = "
            f"{1.0e3 * search_config.disc_radius_m:g} mm\n"
            "Intervals cluster by direction disc "
            f"(df = {search_config.disc_count - 1})"
        )
    else:
        design_note = "Intervals cluster by independent direction disc"
    axis.text(
        0.01,
        0.01,
        design_note,
        transform=axis.transAxes,
        fontsize=8.5,
        color="#334155",
        va="bottom",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _point_row(
    point: RelationshipPoint,
    search: RateCaptureSearchConfig,
    paths: StudyPaths,
    summary: Mapping[str, object],
) -> dict[str, object]:
    loading = summary["loading_rate"]
    if not isinstance(loading, Mapping):
        raise ValueError("point summary is missing its loading-rate payload")
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(f"point {point.slug} is not marked completed")
    if int(summary["sample_count"]) != search.disc_count * search.points_per_disc:
        raise ValueError(f"point {point.slug} has the wrong capture-search count")
    point_search = summary.get("search_config")
    if not isinstance(point_search, Mapping) or point_search.get("phase_space") != "full_sphere":
        raise ValueError(f"point {point.slug} is not a full-sphere run")
    if int(loading["disc_count"]) != search.disc_count:
        raise ValueError(f"point {point.slug} has the wrong direction-cluster count")
    apparatus = metadata["apparatus_config"]
    cooling = apparatus["cooling"]
    config = metadata["multilevel_config"]
    cooling_saturation = metadata["effective_saturation"]["cooling"]
    summary_signature = str(summary["run_signature_sha256"])
    if summary_signature != str(metadata.get("run_signature_sha256", "")):
        raise ValueError(f"point {point.slug} summary and metadata signatures disagree")
    if not np.isclose(
        float(cooling["power_w_per_beam"]),
        point.cooling_power_w_per_beam,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError(f"point {point.slug} cooling power does not match its plan")
    if not np.isclose(
        float(cooling["detuning_hz"]),
        point.cooling_detuning_hz,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(f"point {point.slug} cooling detuning does not match its plan")
    if not np.isclose(
        float(cooling["beam_diameter_m"]), 12.7e-3, rtol=0.0, atol=1.0e-15
    ):
        raise ValueError(f"point {point.slug} did not retain the 12.7 mm cooling beams")
    if not np.isclose(
        float(config["repump_power_w_per_beam"]),
        REPUMP_POWER_W_PER_BEAM,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError(f"point {point.slug} did not retain 0.1 mW repump beams")
    if not np.isclose(
        float(apparatus["repump"]["power_w_per_beam"]),
        REPUMP_POWER_W_PER_BEAM,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError(f"point {point.slug} apparatus repump power is inconsistent")
    actual_s0 = float(
        cooling_saturation["beam_center_on_resonance_saturation_parameter"]
    )
    actual_seff = float(
        cooling_saturation["beam_center_effective_saturation_parameter"]
    )
    if not np.isclose(actual_s0, point.on_resonance_saturation, rtol=2.0e-13, atol=0.0):
        raise ValueError(f"point {point.slug} recorded s0 does not match its plan")
    if not np.isclose(actual_seff, point.effective_saturation, rtol=2.0e-13, atol=0.0):
        raise ValueError(f"point {point.slug} recorded s_eff does not match its plan")
    built_repump = [float(value) for value in metadata["built_repump_beam_powers_w"]]
    if len(built_repump) != 6 or not all(
        np.isclose(value, REPUMP_POWER_W_PER_BEAM, rtol=0.0, atol=1.0e-15)
        for value in built_repump
    ):
        raise ValueError(f"point {point.slug} did not build six 0.1 mW repump components")
    if int(metadata["indexed_state_count"]) != 24 or not bool(config["repumper_enabled"]):
        raise ValueError(f"point {point.slug} did not use the production 24-state repumper model")
    beam_diameter = float(cooling["beam_diameter_m"])
    intensity = 2.0 * point.cooling_power_w_per_beam / (
        pi * (0.5 * beam_diameter) ** 2
    )
    linewidth_rad_per_s = float(config["natural_linewidth_rad_per_s"])
    geometry_hash = str(summary["geometry_sha256"])
    return {
        "point_index": point.point_index,
        "study_key": point.study_key,
        "scan_variable": point.scan_variable,
        "scan_value": point.scan_value,
        "s0": actual_s0,
        "seff": actual_seff,
        "detuning_n": point.cooling_detuning_n,
        "cooling_power_w_per_beam": point.cooling_power_w_per_beam,
        "cooling_power_mw_per_beam": 1.0e3 * point.cooling_power_w_per_beam,
        "cooling_beam_diameter_m": beam_diameter,
        "cooling_beam_diameter_mm": 1.0e3 * beam_diameter,
        "cooling_beam_center_peak_intensity_w_per_m2": intensity,
        "cooling_beam_center_on_resonance_saturation_parameter": actual_s0,
        "cooling_beam_center_effective_saturation_parameter": actual_seff,
        "cooling_detuning_n": point.cooling_detuning_n,
        "cooling_detuning_hz": point.cooling_detuning_hz,
        "cooling_detuning_mhz": point.cooling_detuning_hz / 1.0e6,
        "cooling_detuning_rad_per_s": 2.0 * pi * point.cooling_detuning_hz,
        "natural_linewidth_hz": linewidth_rad_per_s / (2.0 * pi),
        "natural_linewidth_rad_per_s": linewidth_rad_per_s,
        "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
        "repump_power_mw_per_beam": 1.0e3 * REPUMP_POWER_W_PER_BEAM,
        "loading_rate_mean_atoms_per_s": float(loading["loading_rate_mean_atoms_per_s"]),
        "loading_rate_from_mean_spectrum_atoms_per_s": float(
            loading["loading_rate_from_mean_spectrum_atoms_per_s"]
        ),
        "loading_rate_sample_std_atoms_per_s": float(
            loading["loading_rate_sample_std_atoms_per_s"]
        ),
        "loading_rate_disc_cluster_sem_atoms_per_s": float(
            loading["loading_rate_disc_cluster_sem_atoms_per_s"]
        ),
        "loading_rate_t95_lower_atoms_per_s": float(
            loading["loading_rate_t95_lower_atoms_per_s"]
        ),
        "loading_rate_t95_upper_atoms_per_s": float(
            loading["loading_rate_t95_upper_atoms_per_s"]
        ),
        "student_t_critical_95": float(loading["student_t_critical_95"]),
        "confidence_level": float(loading["confidence_level"]),
        "disc_count": search.disc_count,
        "points_per_disc": search.points_per_disc,
        "capture_threshold_search_count": search.disc_count * search.points_per_disc,
        "disc_radius_m": search.disc_radius_m,
        "disc_radius_mm": 1.0e3 * search.disc_radius_m,
        "phase_space": search.phase_space,
        "geometry_sha256": geometry_hash,
        "run_signature_sha256": summary_signature,
        "point_elapsed_wall_time_s": float(metadata.get("elapsed_wall_time_s", 0.0)),
        "statistics_directory": str(paths.statistics.resolve()),
        "figures_directory": str(paths.figures.resolve()),
        "capture_cross_section_csv": str(paths.spectrum_csv.resolve()),
        "capture_cross_section_plot": str(paths.cross_section_png.resolve()),
        "status": "completed",
    }


def _study_metadata(
    study_key: str,
    points: Sequence[RelationshipPoint],
    rows: Sequence[Mapping[str, object]],
    search: RateCaptureSearchConfig,
    common_geometry_hash: str,
    paths: CampaignPaths,
    *,
    status: str,
) -> dict[str, object]:
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "study_key": study_key,
        "status": status,
        "updated_utc": _utc_now(),
        "model": "24-state repumper-included multilevel population-rate MOT",
        "point_count_requested": len(points),
        "point_count_completed": len(rows),
        "ordered_point_plan": [asdict(point) for point in points],
        "search_config": asdict(search),
        "common_geometry_sha256": common_geometry_hash,
        "common_random_numbers": (
            "The same seeded full-sphere direction discs and uniform-area point coordinates "
            "are reused at every parameter point; only the named scan variable changes."
        ),
        "uncertainty": (
            f"Each point is the arithmetic mean of {search.disc_count} direction-disc "
            "loading estimates. Error bars are two-sided 95% Student-t intervals "
            f"across those {search.disc_count} independent direction clusters "
            f"(df={search.disc_count - 1}), not across "
            f"{search.disc_count * search.points_per_disc} launch points."
        ),
        "statistics_csv": str(paths.aggregate_csv(study_key).resolve()),
        "relationship_plot": str(paths.relationship_plot(study_key).resolve()),
    }


def run_loading_relationship(
    study_key: str,
    points: Sequence[RelationshipPoint],
    *,
    search_config: RateCaptureSearchConfig,
    paths: CampaignPaths,
    worker_count: int,
    resume: bool,
    plot_only: bool = False,
    overall_offset: int = 0,
    overall_total: int = 46,
) -> list[dict[str, object]]:
    """Run one complete relationship before allowing the next one to start."""

    if search_config.phase_space != "full_sphere":
        raise ValueError("relationship campaigns require full-sphere direction sampling")
    if (
        search_config.disc_count < 2
        or search_config.points_per_disc < 1
        or not np.isclose(search_config.disc_radius_m, DEFAULT_DISC_RADIUS_M)
        or search_config.include_center_point
    ):
        raise ValueError(
            "production relationship campaigns require at least two random full-sphere "
            "direction discs, at least one point per disc, and r=15 mm geometry"
        )
    common_hash = _geometry_hash(search_config)
    rows: list[dict[str, object]] = []
    study_start = perf_counter()
    for local_index, point in enumerate(points, start=1):
        point_paths = paths.point_paths(point)
        overall_index = overall_offset + local_index
        sample_scale = (
            search_config.disc_count
            * search_config.points_per_disc
            / (SATURATION_DISC_COUNT * SATURATION_POINTS_PER_DISC)
        )
        estimated_remaining = REFERENCE_SECONDS_PER_LOADING_POINT * sample_scale * (
            overall_total - overall_index + 1
        )
        print(
            f"[{_timestamp()}] [campaign {overall_index}/{overall_total}] "
            f"[{study_key} {local_index}/{len(points)}] starting {point.slug}; "
            f"P={1e3 * point.cooling_power_w_per_beam:.9g} mW, "
            f"Delta/Gamma={point.cooling_detuning_n:.9g}; "
            f"planning ETA remaining={estimated_remaining / 3600.0:.2f} h",
            flush=True,
        )
        if plot_only:
            if not point_paths.capture_summary_json.is_file():
                raise FileNotFoundError(
                    f"plot-only requested but point summary is missing: {point_paths.capture_summary_json}"
                )
            summary = json.loads(point_paths.capture_summary_json.read_text(encoding="utf-8"))
        else:
            complete = False
            if resume and point_paths.metadata_json.is_file():
                prior = json.loads(point_paths.metadata_json.read_text(encoding="utf-8"))
                complete = (
                    prior.get("status") == "completed"
                    and int(prior.get("completed_sample_count", -1))
                    == search_config.disc_count * search_config.points_per_disc
                    and point_paths.final_samples_csv.is_file()
                )
            summary = run_power_loading_study(
                search_config=search_config,
                worker_count=worker_count,
                output_directory=point_paths.statistics,
                figure_directory=point_paths.figures,
                resume=resume,
                analyze_only=complete,
                cooling_power_w_per_beam=point.cooling_power_w_per_beam,
                repump_power_w_per_beam=REPUMP_POWER_W_PER_BEAM,
                cooling_detuning_hz=point.cooling_detuning_hz,
                study_name=f"{paths.statistics.name}/{study_key}/{point.slug}",
                progress_every=DEFAULT_PROGRESS_EVERY,
                checkpoint_every=DEFAULT_CHECKPOINT_EVERY,
                plot_context=(
                    f"s0={point.on_resonance_saturation:.6g}; "
                    f"seff={point.effective_saturation:.6g}; "
                    f"P={1e3 * point.cooling_power_w_per_beam:.6g} mW; "
                    f"Delta/Gamma={point.cooling_detuning_n:.6g}; "
                    "0.1 mW repump; full sphere; r=15 mm"
                ),
            )
        row = _point_row(point, search_config, point_paths, summary)
        if row["geometry_sha256"] != common_hash:
            raise ValueError(f"point {point.slug} did not reuse the common seeded geometry")
        rows.append(row)
        _atomic_write_text(paths.aggregate_csv(study_key), _csv_text(rows))
        plot_loading_relationship(
            rows,
            study_key,
            paths.relationship_plot(study_key),
            search_config=search_config,
        )
        _atomic_write_json(
            paths.study_metadata_json(study_key),
            _study_metadata(
                study_key,
                points,
                rows,
                search_config,
                common_hash,
                paths,
                status="completed" if len(rows) == len(points) else "running",
            ),
        )
        point_elapsed = float(row["point_elapsed_wall_time_s"])
        completed_wall = sum(float(item["point_elapsed_wall_time_s"]) for item in rows)
        empirical_per_point = completed_wall / len(rows) if completed_wall > 0.0 else REFERENCE_SECONDS_PER_LOADING_POINT
        remaining_points = overall_total - overall_index
        eta = empirical_per_point * remaining_points
        print(
            f"[{_timestamp()}] [campaign {overall_index}/{overall_total}] completed "
            f"{point.slug} in {point_elapsed / 60.0:.2f} min; "
            f"R={float(row['loading_rate_mean_atoms_per_s']):.6g} atoms/s; "
            f"aggregate and plot saved; estimated loading ETA={eta / 3600.0:.2f} h",
            flush=True,
        )
    if len(rows) == len(points):
        elapsed = perf_counter() - study_start
        print(
            f"[{_timestamp()}] {study_key} fully completed: {len(points)} points "
            f"in this invocation's {elapsed / 3600.0:.2f} h wall time",
            flush=True,
        )
    return rows


def run_relationship_loading_campaign(
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    search_config: RateCaptureSearchConfig | None = None,
    paths: CampaignPaths | None = None,
    resume: bool = True,
    plot_only: bool = False,
    selected_studies: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run the selected loading studies serially, with no parameter convolution."""

    if worker_count <= 0 or worker_count > 24:
        raise ValueError("worker_count must be in the range 1..24")
    groups = requested_relationship_points()
    ordered_keys = (RAW_STUDY_KEY, EFFECTIVE_STUDY_KEY, DETUNING_STUDY_KEY)
    chosen = tuple(selected_studies) if selected_studies is not None else ordered_keys
    if any(key not in ordered_keys for key in chosen):
        raise ValueError("selected_studies contains an unknown study")
    chosen = tuple(key for key in ordered_keys if key in chosen)
    if not chosen:
        raise ValueError("selected_studies must contain at least one study")
    saturation_only = set(chosen).issubset({RAW_STUDY_KEY, EFFECTIVE_STUDY_KEY})
    detuning_only = chosen == (DETUNING_STUDY_KEY,)
    if paths is None and not (saturation_only or detuning_only):
        raise ValueError(
            "mixed saturation and detuning studies require explicit paths; the production "
            "campaign stores them in separate 30x30 and 15x15 roots"
        )
    if search_config is None:
        search = (
            default_saturation_search_config(worker_count=worker_count)
            if saturation_only
            else default_relationship_search_config(worker_count=worker_count)
        )
    else:
        search = search_config
    if paths is None:
        campaign_paths = (
            default_campaign_paths()
            if saturation_only
            else default_remaining_campaign_paths()
        )
    else:
        campaign_paths = paths
    selected_groups = {key: groups[key] for key in chosen}
    total = sum(len(points) for points in selected_groups.values())
    common_hash = _geometry_hash(search)
    signature = _campaign_signature(search, groups)
    campaign_paths.statistics.mkdir(parents=True, exist_ok=True)
    campaign_paths.figures.mkdir(parents=True, exist_ok=True)
    started_utc = _utc_now()
    if resume and campaign_paths.metadata_json.is_file():
        prior = json.loads(campaign_paths.metadata_json.read_text(encoding="utf-8"))
        if prior.get("campaign_signature_sha256") != signature:
            raise ValueError("campaign resume signature mismatch")
        started_utc = str(prior.get("started_utc", started_utc))
    metadata: dict[str, object] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_name": campaign_paths.statistics.name,
        "campaign_signature_sha256": signature,
        "status": "running",
        "started_utc": started_utc,
        "updated_utc": _utc_now(),
        "model": "24-state repumper-included multilevel population-rate MOT",
        "execution_order": list(chosen),
        "selected_studies": list(chosen),
        "loading_point_count_in_full_campaign": total,
        "capture_threshold_search_count_in_full_campaign": (
            total * search.disc_count * search.points_per_disc
        ),
        "combinatorial_product_used": False,
        "search_config": asdict(search),
        "common_geometry_sha256": common_hash,
        "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
        "reference_cooling_power_w_per_beam": DEFAULT_COOLING_POWER_W_PER_BEAM,
        "baseline_cooling_detuning_hz": COOLING_DETUNING_HZ,
        "definitions": {
            "s0": "I0/I_sat for one Gaussian cooling beam at its center, I0=2P/(pi*w^2)",
            "s_eff": "s0/[1+(2*Delta/Gamma)^2] for one cooling beam at its center",
            "detuning_n": "signed angular-frequency ratio Delta/Gamma; negative is red detuning",
        },
        "ordered_grids": {
            key: [point.scan_value for point in groups[key]] for key in chosen
        },
        "statistics_root": str(campaign_paths.statistics.resolve()),
        "figures_root": str(campaign_paths.figures.resolve()),
    }
    _atomic_write_json(campaign_paths.metadata_json, metadata)
    completed = 0
    results: dict[str, object] = {}
    try:
        for key in chosen:
            rows = run_loading_relationship(
                key,
                groups[key],
                search_config=search,
                paths=campaign_paths,
                worker_count=worker_count,
                resume=resume,
                plot_only=plot_only,
                overall_offset=completed,
                overall_total=total,
            )
            completed += len(rows)
            results[key] = {
                "row_count": len(rows),
                "aggregate_csv": str(campaign_paths.aggregate_csv(key).resolve()),
                "plot": str(campaign_paths.relationship_plot(key).resolve()),
            }
            metadata.update(
                {
                    "updated_utc": _utc_now(),
                    "completed_selected_loading_point_count": completed,
                    "selected_loading_point_count": total,
                    "last_completed_study": key,
                    "results": results,
                }
            )
            _atomic_write_json(campaign_paths.metadata_json, metadata)
    except BaseException as exc:
        metadata.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "updated_utc": _utc_now(),
                "completed_selected_loading_point_count": completed,
                "selected_loading_point_count": total,
                "last_error": f"{type(exc).__name__}: {exc}",
                "results": results,
            }
        )
        _atomic_write_json(campaign_paths.metadata_json, metadata)
        raise
    metadata.update(
        {
            "status": "completed",
            "updated_utc": _utc_now(),
            "completed_selected_loading_point_count": completed,
            "selected_loading_point_count": total,
            "results": results,
        }
    )
    _atomic_write_json(campaign_paths.metadata_json, metadata)
    print(
        f"[{_timestamp()}] selected loading campaign completed: {completed}/{total} points; "
        f"data root={campaign_paths.statistics.resolve()}; figures root={campaign_paths.figures.resolve()}",
        flush=True,
    )
    return metadata


def run_relationship_temperature_campaign(
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    paths: CampaignPaths | None = None,
    resume: bool = True,
    plot_only: bool = False,
) -> dict[str, object]:
    """Run the physically separate 15-cloud × 15-atom detuning temperature study.

    Launch discs estimate capture/loading.  Temperature instead uses independent
    preloaded clouds and Langevin recoil diffusion, with the same initial clouds
    reused at every detuning.  This avoids conflating capture with thermalization.
    """

    if worker_count <= 0 or worker_count > 24:
        raise ValueError("worker_count must be in the range 1..24")
    campaign_paths = paths or default_remaining_campaign_paths()
    summary_csv = campaign_paths.temperature_statistics / "temperature_vs_detuning.csv"
    metadata_json = (
        campaign_paths.temperature_statistics / "temperature_vs_detuning_metadata.json"
    )
    combined_plot = campaign_paths.temperature_figures / "temperature_vs_detuning.png"
    temperature_only_plot = (
        campaign_paths.temperature_figures / "temperature_only_vs_detuning.png"
    )
    print(
        f"[{_timestamp()}] [temperature 1/{len(DETUNING_N_VALUES)} through "
        f"{len(DETUNING_N_VALUES)}/{len(DETUNING_N_VALUES)}] starting "
        f"{TEMPERATURE_ENSEMBLE_COUNT} preloaded clouds × "
        f"{TEMPERATURE_ATOMS_PER_ENSEMBLE} atoms at each detuning; "
        "27 mW cooling; 0.1 mW repump; 25 ms trajectories",
        flush=True,
    )
    if plot_only:
        validated_metadata = _validate_temperature_campaign_products(
            summary_csv, metadata_json
        )
        plot_temperature_vs_detuning(
            summary_csv,
            combined_plot,
            cooling_power_w_per_beam=DEFAULT_COOLING_POWER_W_PER_BEAM,
            ensemble_realization_count=TEMPERATURE_ENSEMBLE_COUNT,
            atoms_per_ensemble=TEMPERATURE_ATOMS_PER_ENSEMBLE,
            include_survivor_panel=True,
        )
        plot_temperature_vs_detuning(
            summary_csv,
            temperature_only_plot,
            cooling_power_w_per_beam=DEFAULT_COOLING_POWER_W_PER_BEAM,
            ensemble_realization_count=TEMPERATURE_ENSEMBLE_COUNT,
            atoms_per_ensemble=TEMPERATURE_ATOMS_PER_ENSEMBLE,
            include_survivor_panel=False,
        )
        result: dict[str, object] = {
            "status": "completed",
            "completed_point_count": len(DETUNING_N_VALUES),
            "total_point_count": len(DETUNING_N_VALUES),
            "outputs": {
                "temperature_summary_csv": str(summary_csv.resolve()),
                "temperature_plot": str(combined_plot.resolve()),
                "temperature_only_plot": str(temperature_only_plot.resolve()),
                "metadata_json": str(metadata_json.resolve()),
            },
        }
    else:
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
    if result.get("status") != "completed" or int(result["completed_point_count"]) != len(
        DETUNING_N_VALUES
    ):
        raise RuntimeError("temperature sweep did not complete all requested detunings")
    validated_metadata = _validate_temperature_campaign_products(
        summary_csv, metadata_json
    )
    print(
        f"[{_timestamp()}] temperature campaign completed; "
        f"data={summary_csv.resolve()}; plot={temperature_only_plot.resolve()}",
        flush=True,
    )
    return result


def _validate_temperature_campaign_products(
    summary_csv: Path,
    metadata_json: Path,
) -> dict[str, object]:
    """Reject incomplete or foreign temperature products before plotting/certifying."""

    if not summary_csv.is_file() or not metadata_json.is_file():
        raise FileNotFoundError(
            "temperature campaign requires both its summary CSV and signed metadata"
        )
    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError("temperature metadata is not marked completed")
    if int(metadata.get("completed_point_count", -1)) != len(DETUNING_N_VALUES):
        raise ValueError("temperature metadata has the wrong completed-point count")
    if int(metadata.get("completed_ensemble_row_count", -1)) != (
        len(DETUNING_N_VALUES) * TEMPERATURE_ENSEMBLE_COUNT
    ):
        raise ValueError("temperature metadata has the wrong cloud-row count")
    signature = metadata.get("resume_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("temperature metadata is missing its resume signature")
    if signature.get("solver") != (
        "24_state_repumper_adiabatic_population_rate_equation_langevin"
    ):
        raise ValueError("temperature products did not use the production Langevin solver")
    if int(signature.get("ensemble_realization_count", -1)) != TEMPERATURE_ENSEMBLE_COUNT:
        raise ValueError("temperature products have the wrong independent-cloud count")
    if int(signature.get("atoms_per_ensemble", -1)) != TEMPERATURE_ATOMS_PER_ENSEMBLE:
        raise ValueError("temperature products have the wrong atoms-per-cloud count")
    if int(signature.get("trajectory_count_per_point", -1)) != (
        TEMPERATURE_ENSEMBLE_COUNT * TEMPERATURE_ATOMS_PER_ENSEMBLE
    ):
        raise ValueError("temperature products have the wrong trajectory count")
    recorded_detunings = np.asarray(signature.get("detuning_n_values", []), dtype=float)
    if recorded_detunings.shape != (len(DETUNING_N_VALUES),) or not np.allclose(
        recorded_detunings,
        np.asarray(DETUNING_N_VALUES),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("temperature products use the wrong detuning grid or order")
    if not np.isclose(
        float(signature.get("cooling_power_w_per_beam", np.nan)),
        DEFAULT_COOLING_POWER_W_PER_BEAM,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("temperature products did not use 27 mW cooling beams")
    config = signature.get("multilevel_config")
    apparatus = signature.get("apparatus_config")
    if not isinstance(config, Mapping) or not bool(config.get("repumper_enabled")):
        raise ValueError("temperature products did not enable the 24-state repumper model")
    if not np.isclose(
        float(config.get("repump_power_w_per_beam", np.nan)),
        REPUMP_POWER_W_PER_BEAM,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("temperature solver metadata did not retain 0.1 mW repump")
    if not isinstance(apparatus, Mapping) or not np.isclose(
        float(apparatus["repump"]["power_w_per_beam"]),
        REPUMP_POWER_W_PER_BEAM,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("temperature apparatus metadata did not retain 0.1 mW repump")
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(DETUNING_N_VALUES):
        raise ValueError("temperature summary CSV has the wrong row count")
    for expected_n, row in zip(DETUNING_N_VALUES, rows, strict=True):
        if not np.isclose(float(row["detuning_n"]), expected_n, rtol=0.0, atol=1.0e-12):
            raise ValueError("temperature summary CSV has the wrong detuning order")
        if int(row["requested_ensemble_count"]) != TEMPERATURE_ENSEMBLE_COUNT:
            raise ValueError("temperature summary CSV has the wrong cloud count")
        if int(row["requested_atom_count"]) != (
            TEMPERATURE_ENSEMBLE_COUNT * TEMPERATURE_ATOMS_PER_ENSEMBLE
        ):
            raise ValueError("temperature summary CSV has the wrong atom count")
        if not np.isclose(
            float(row["cooling_power_w_per_beam"]),
            DEFAULT_COOLING_POWER_W_PER_BEAM,
            rtol=0.0,
            atol=1.0e-15,
        ):
            raise ValueError("temperature summary CSV has the wrong cooling power")
    return metadata


def run_relationship_campaign(
    *,
    worker_count: int = DEFAULT_WORKER_COUNT,
    search_config: RateCaptureSearchConfig | None = None,
    paths: CampaignPaths | None = None,
    resume: bool = True,
    plot_only: bool = False,
) -> dict[str, object]:
    """Run only the remaining 15x15 detuning loading and temperature studies.

    The completed raw- and effective-saturation products remain untouched in
    the separate 30x30 saturation root returned by :func:`default_campaign_paths`.
    """

    campaign_paths = paths or default_remaining_campaign_paths()
    remaining_search = search_config or default_relationship_search_config(
        worker_count=worker_count
    )
    loading = run_relationship_loading_campaign(
        worker_count=worker_count,
        search_config=remaining_search,
        paths=campaign_paths,
        resume=resume,
        plot_only=plot_only,
        selected_studies=(DETUNING_STUDY_KEY,),
    )
    metadata = json.loads(campaign_paths.metadata_json.read_text(encoding="utf-8"))
    metadata.update(
        {
            "status": "temperature_running",
            "updated_utc": _utc_now(),
            "temperature_design": {
                "ensemble_realization_count": TEMPERATURE_ENSEMBLE_COUNT,
                "atoms_per_ensemble": TEMPERATURE_ATOMS_PER_ENSEMBLE,
                "trajectory_count_per_detuning": (
                    TEMPERATURE_ENSEMBLE_COUNT * TEMPERATURE_ATOMS_PER_ENSEMBLE
                ),
                "detuning_point_count": len(DETUNING_N_VALUES),
                "cooling_power_w_per_beam": DEFAULT_COOLING_POWER_W_PER_BEAM,
                "repump_power_w_per_beam": REPUMP_POWER_W_PER_BEAM,
                "interpretation": (
                    "Temperature uses preloaded Langevin clouds, not incident launch discs; "
                    "only cooling detuning changes between points."
                ),
            },
        }
    )
    _atomic_write_json(campaign_paths.metadata_json, metadata)
    try:
        temperature = run_relationship_temperature_campaign(
            worker_count=worker_count,
            paths=campaign_paths,
            resume=resume,
            plot_only=plot_only,
        )
    except BaseException as exc:
        metadata.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "updated_utc": _utc_now(),
                "last_error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(campaign_paths.metadata_json, metadata)
        raise
    metadata.update(
        {
            "status": "completed",
            "updated_utc": _utc_now(),
            "temperature_result": {
                "status": temperature["status"],
                "completed_point_count": temperature["completed_point_count"],
                "outputs": temperature["outputs"],
            },
        }
    )
    _atomic_write_json(campaign_paths.metadata_json, metadata)
    saturation_paths = default_campaign_paths()
    return {
        "status": "completed",
        "preserved_saturation_outputs": {
            "statistics_root": str(saturation_paths.statistics.resolve()),
            "figures_root": str(saturation_paths.figures.resolve()),
        },
        "loading": loading,
        "temperature": temperature,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent 24-state MOT loading-relationship studies"
    )
    parser.add_argument(
        "study",
        choices=(
            "campaign",
            "raw-s",
            "effective-s",
            "detuning-loading",
            "temperature",
        ),
        nargs="?",
        default="campaign",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKER_COUNT)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--plot-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.study == "campaign":
        run_relationship_campaign(
            worker_count=args.workers,
            resume=args.resume,
            plot_only=args.plot_only,
        )
        return 0
    if args.study == "temperature":
        run_relationship_temperature_campaign(
            worker_count=args.workers,
            resume=args.resume,
            plot_only=args.plot_only,
        )
        return 0
    selections = {
        "raw-s": (RAW_STUDY_KEY,),
        "effective-s": (EFFECTIVE_STUDY_KEY,),
        "detuning-loading": (DETUNING_STUDY_KEY,),
    }
    run_relationship_loading_campaign(
        worker_count=args.workers,
        resume=args.resume,
        plot_only=args.plot_only,
        selected_studies=selections[args.study],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_NAME",
    "CampaignPaths",
    "DEFAULT_DISC_COUNT",
    "DEFAULT_DISC_RADIUS_M",
    "DEFAULT_POINTS_PER_DISC",
    "DETUNING_N_VALUES",
    "DETUNING_STUDY_KEY",
    "EFFECTIVE_SATURATION_VALUES",
    "EFFECTIVE_STUDY_KEY",
    "REMAINING_CAMPAIGN_NAME",
    "RAW_SATURATION_VALUES",
    "RAW_STUDY_KEY",
    "RelationshipPoint",
    "SATURATION_CAMPAIGN_NAME",
    "SATURATION_DISC_COUNT",
    "SATURATION_POINTS_PER_DISC",
    "build_relationship_points",
    "default_campaign_paths",
    "default_remaining_campaign_paths",
    "default_relationship_search_config",
    "default_saturation_search_config",
    "detuning_reduction_denominator",
    "effective_saturation_from_s0",
    "on_resonance_saturation_parameter",
    "plot_loading_relationship",
    "requested_relationship_points",
    "run_loading_relationship",
    "run_relationship_campaign",
    "run_relationship_loading_campaign",
    "run_relationship_temperature_campaign",
    "saturation_power_w_per_beam",
]
