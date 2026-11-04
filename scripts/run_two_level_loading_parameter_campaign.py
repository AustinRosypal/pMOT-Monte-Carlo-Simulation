"""Complete and plot five effective two-level MOT loading-rate scans.

All five relationships use the current corrected ordinary-frequency force law
and the same 25 by 25 full-sphere launch geometry plus fail-closed endpoint-
audit machinery.  The September 2026 pre-prefactor-correction products remain
immutable provenance but are not used as current simulation results.

The beam-diameter scan holds the Gaussian center saturation fixed, so its power
scales as diameter squared.  It is run only after the gradient scan and uses
the sampled gradient with the largest mean loading rate (ties choose the lower
gradient).  The plotting stage treats direction discs, not individual rays,
as the independent clusters and draws the saved two-sided 95% Student-t
interval as a translucent band.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from math import pi
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_simple import refined_relationship_campaign as established
from pmot.mot_simple.configuration import (
    default_simple_mot_apparatus,
    default_simple_mot_config,
)
from pmot.mot_simple.power_loading_study import StudyPaths, geometry_csv_text, geometry_rows
from pmot.mot_simple.simulation import build_simple_mot_beams


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_NAME = "loading_parameter_scans_25x25_20261103"
STATISTICS_ROOT = PROJECT_ROOT / "outputs" / "statistics" / "mot_simple" / CAMPAIGN_NAME
FIGURES_ROOT = PROJECT_ROOT / "outputs" / "figures" / "mot_simple" / CAMPAIGN_NAME
ESTABLISHED_NAME = established.CAMPAIGN_NAME
ESTABLISHED_STATISTICS = (
    PROJECT_ROOT / "outputs" / "statistics" / "mot_simple" / ESTABLISHED_NAME
)

GRADIENT_STUDY_KEY = "04_magnetic_gradient"
DIAMETER_STUDY_KEY = "05_beam_diameter"
GRADIENTS_G_PER_CM = (10.0, 15.0, 20.0, 25.0, 30.0)
DIAMETERS_MM = (5.0, 7.0, 10.0, 12.7, 15.0, 18.0, 21.0, 25.0)
OPTIMAL_S0 = 70.0
OPTIMAL_DETUNING_N = -2.25
REFERENCE_DIAMETER_MM = 12.7
WORKER_LIMIT = 24

SCAN_DEFINITIONS = {
    established.RAW_STUDY_KEY: {
        "source": "new",
        "xfield": "s0",
        "xlabel": r"On-resonance single-beam saturation $s_0$",
        "filename": "loading_rate_vs_saturation.png",
        "title": "Two-Level MOT Loading Rate vs Saturation",
        "fixed": "Δ/Γ = −2.25 (−13.646 MHz); Gz = 10 G/cm; D = 12.7 mm; power varies",
    },
    established.EFFECTIVE_STUDY_KEY: {
        "source": "new",
        "xfield": "seff",
        "xlabel": r"Effective single-beam saturation $s_{\mathrm{eff}}$",
        "filename": "loading_rate_vs_effective_saturation.png",
        "title": "Two-Level MOT Loading Rate vs Effective Saturation",
        "fixed": "Δ/Γ = −2.25 (−13.646 MHz); Gz = 10 G/cm; D = 12.7 mm; power varies",
    },
    established.DETUNING_STUDY_KEY: {
        "source": "new",
        "xfield": "detuning_n",
        "xlabel": r"Normalized angular detuning $\Delta_\omega/\Gamma$",
        "filename": "loading_rate_vs_detuning.png",
        "title": "Two-Level MOT Loading Rate vs Detuning",
        "fixed": "s₀ selected from corrected saturation scan; Gz = 10 G/cm; D = 12.7 mm",
    },
    GRADIENT_STUDY_KEY: {
        "source": "new",
        "xfield": "gradient_g_per_cm",
        "xlabel": r"Axial magnetic-field gradient $G_z$ [G/cm]",
        "filename": "loading_rate_vs_magnetic_gradient.png",
        "title": "Two-Level MOT Loading Rate vs Magnetic-Field Gradient",
        "fixed": "s₀ = 70; Δ/Γ = −2.25 (−13.646 MHz); D = 12.7 mm; P = 73.998 mW/beam",
    },
    DIAMETER_STUDY_KEY: {
        "source": "new",
        "xfield": "beam_diameter_mm",
        "xlabel": r"Cooling Gaussian $1/e^2$ beam diameter $D$ [mm]",
        "filename": "loading_rate_vs_beam_diameter.png",
        "title": "Two-Level MOT Loading Rate vs Beam Size",
        "fixed": "s₀ = 70; Δ/Γ = −2.25 (−13.646 MHz); P(D) ∝ D²; gradient selected from scan",
    },
}

_ACTIVE_DIAMETER_GRADIENT_G_PER_CM = 10.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _point_paths(point: established.RelationshipPoint) -> StudyPaths:
    return StudyPaths(
        statistics=STATISTICS_ROOT / point.study_key / "points" / point.slug,
        figures=FIGURES_ROOT / point.study_key / "points" / point.slug,
    )


def _diameter_mm(point: established.RelationshipPoint) -> float:
    return point.scan_value if point.study_key == DIAMETER_STUDY_KEY else REFERENCE_DIAMETER_MM


def _gradient_g_per_cm(point: established.RelationshipPoint) -> float:
    if point.study_key == GRADIENT_STUDY_KEY:
        return point.scan_value
    if point.study_key == DIAMETER_STUDY_KEY:
        return _ACTIVE_DIAMETER_GRADIENT_G_PER_CM
    return 10.0


def _build_extended_point_configuration(point: established.RelationshipPoint):
    """Build configurations for the two added scan families.

    This function is installed only in the parent campaign process.  Spawned
    workers receive the fully materialized apparatus, atom, coil, and search
    configurations through the established worker initializer.
    """

    diameter_m = 1.0e-3 * _diameter_mm(point)
    apparatus = default_simple_mot_apparatus()
    apparatus = replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            power_w_per_beam=point.cooling_power_w_per_beam,
            detuning_hz=point.cooling_detuning_hz,
            beam_diameter_m=diameter_m,
        ),
    )
    simple = replace(
        default_simple_mot_config(), cooling_detuning_hz=point.cooling_detuning_hz
    )
    coil = default_anti_helmholtz_config(
        target_gradient_g_per_cm=_gradient_g_per_cm(point)
    )
    beams = build_simple_mot_beams(apparatus, simple)
    if len(beams) != 6:
        raise RuntimeError("the effective two-level campaign requires six cooling beams")
    if not all(
        np.isclose(beam.intensity_beam.power_w, point.cooling_power_w_per_beam)
        and np.isclose(2.0 * beam.intensity_beam.beam_radius_m, diameter_m)
        and np.isclose(beam.detuning_hz, point.cooling_detuning_hz)
        for beam in beams
    ):
        raise RuntimeError("built cooling beams do not reproduce the extended point")
    return apparatus, simple, coil, beams


def _points(study_key: str) -> tuple[established.RelationshipPoint, ...]:
    simple = default_simple_mot_config()
    fixed_detuning_hz = OPTIMAL_DETUNING_N * simple.linewidth_hz
    if study_key == established.RAW_STUDY_KEY:
        values = established.RAW_SATURATION_VALUES
    elif study_key == established.EFFECTIVE_STUDY_KEY:
        values = established.EFFECTIVE_SATURATION_VALUES
    elif study_key == established.DETUNING_STUDY_KEY:
        values = established.DETUNING_N_VALUES
    elif study_key == GRADIENT_STUDY_KEY:
        values = GRADIENTS_G_PER_CM
    elif study_key == DIAMETER_STUDY_KEY:
        values = DIAMETERS_MM
    else:
        raise ValueError(f"unknown study {study_key}")
    result = []
    for index, value in enumerate(values):
        diameter_m = 1.0e-3 * (
            value if study_key == DIAMETER_STUDY_KEY else REFERENCE_DIAMETER_MM
        )
        if study_key == established.RAW_STUDY_KEY:
            s0 = float(value)
            detuning_n = OPTIMAL_DETUNING_N
            detuning_hz = fixed_detuning_hz
            scan_variable = "s0"
        elif study_key == established.EFFECTIVE_STUDY_KEY:
            detuning_n = OPTIMAL_DETUNING_N
            detuning_hz = fixed_detuning_hz
            s0 = float(value) * established.detuning_reduction_denominator(detuning_n)
            scan_variable = "seff"
        elif study_key == established.DETUNING_STUDY_KEY:
            s0 = _best_s0()
            detuning_n = float(value)
            detuning_hz = detuning_n * simple.linewidth_hz
            scan_variable = "detuning_n"
        else:
            s0 = OPTIMAL_S0
            detuning_n = OPTIMAL_DETUNING_N
            detuning_hz = fixed_detuning_hz
            scan_variable = (
                "gradient_g_per_cm"
                if study_key == GRADIENT_STUDY_KEY
                else "beam_diameter_mm"
            )
        power_w = established.saturation_power_w_per_beam(s0, beam_diameter_m=diameter_m)
        result.append(
            established.RelationshipPoint(
                study_key=study_key,
                point_index=index,
                scan_variable=scan_variable,
                scan_value=float(value),
                cooling_power_w_per_beam=power_w,
                cooling_detuning_n=detuning_n,
                cooling_detuning_hz=detuning_hz,
                on_resonance_saturation=s0,
                effective_saturation=established.effective_saturation_from_s0(
                    s0, detuning_n
                ),
            )
        )
    return tuple(result)


def _geometry():
    search = established.default_search_config()
    discs, points = established.generate_common_geometry(search)
    text = geometry_csv_text(geometry_rows(discs, points))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return search, discs, points, text, digest


def _result_row(point: established.RelationshipPoint, summary: Mapping[str, object]) -> dict[str, object]:
    loading = summary["loading_rate"]
    if not isinstance(loading, Mapping):
        raise ValueError("completed point has no loading-rate result")
    paths = _point_paths(point)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or int(summary["unresolved_timeout_count"]) != 0:
        raise ValueError(f"point {point.slug} is not a complete fail-closed result")
    return {
        "point_index": point.point_index,
        "study_key": point.study_key,
        "scan_value": point.scan_value,
        "gradient_g_per_cm": _gradient_g_per_cm(point),
        "beam_diameter_mm": _diameter_mm(point),
        "cooling_power_mw_per_beam": 1.0e3 * point.cooling_power_w_per_beam,
        "s0": point.on_resonance_saturation,
        "seff": point.effective_saturation,
        "detuning_n": point.cooling_detuning_n,
        "cooling_detuning_mhz": point.cooling_detuning_hz / 1.0e6,
        "loading_rate_mean_atoms_per_s": loading["loading_rate_mean_atoms_per_s"],
        "loading_rate_disc_cluster_sem_atoms_per_s": loading[
            "loading_rate_disc_cluster_sem_atoms_per_s"
        ],
        "loading_rate_t95_lower_atoms_per_s": loading[
            "loading_rate_t95_lower_atoms_per_s"
        ],
        "loading_rate_t95_upper_atoms_per_s": loading[
            "loading_rate_t95_upper_atoms_per_s"
        ],
        "student_t_critical_95": loading["student_t_critical_95"],
        "confidence_level": loading["confidence_level"],
        "disc_count": 25,
        "points_per_disc": 25,
        "disc_radius_mm": 15.0,
        "phase_space": "full_sphere",
        "geometry_sha256": summary["geometry_sha256"],
        "velocity_resolved_override_count": summary[
            "velocity_resolved_override_count"
        ],
        "indeterminate_zero_flux_ray_count": summary[
            "indeterminate_zero_flux_ray_count"
        ],
        "unresolved_timeout_count": summary["unresolved_timeout_count"],
        "point_elapsed_wall_time_s": metadata["elapsed_wall_time_s"],
        "statistics_directory": str(paths.statistics.resolve()),
        "status": "completed",
    }


def run_study(study_key: str, *, workers: int, resume: bool) -> list[dict[str, object]]:
    if study_key not in SCAN_DEFINITIONS:
        raise ValueError(f"unsupported new study {study_key}")
    search, discs, launch_points, geometry_text, geometry_hash = _geometry()
    STATISTICS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    geometry_path = STATISTICS_ROOT / "launch_geometry.csv"
    if geometry_path.exists():
        if hashlib.sha256(geometry_path.read_bytes()).hexdigest() != geometry_hash:
            raise ValueError("saved common launch geometry changed")
    else:
        geometry_path.write_text(geometry_text, encoding="utf-8", newline="")

    prior_builder = established.build_point_configuration
    prior_name = established.CAMPAIGN_NAME
    established.build_point_configuration = _build_extended_point_configuration
    established.CAMPAIGN_NAME = CAMPAIGN_NAME
    rows: list[dict[str, object]] = []
    try:
        for point in _points(study_key):
            paths = _point_paths(point)
            complete = False
            if resume and paths.metadata_json.exists() and paths.final_samples_csv.exists():
                prior = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
                complete = (
                    prior.get("status") == "completed"
                    and int(prior.get("completed_sample_count", -1)) == 625
                )
            print(
                f"[{_utc_now()}] starting {point.study_key}/{point.slug}; "
                f"Gz={_gradient_g_per_cm(point):g} G/cm, D={_diameter_mm(point):g} mm, "
                f"P={1e3 * point.cooling_power_w_per_beam:.6g} mW/beam",
                flush=True,
            )
            summary = established.run_relationship_point(
                point,
                search=search,
                discs=discs,
                points=launch_points,
                geometry_text=geometry_text,
                geometry_hash=geometry_hash,
                paths=paths,
                worker_count=workers,
                resume=resume,
                analyze_only=complete,
            )
            rows.append(_result_row(point, summary))
            _write_csv(STATISTICS_ROOT / study_key / "aggregate.csv", rows)
    finally:
        established.build_point_configuration = prior_builder
        established.CAMPAIGN_NAME = prior_name
    return rows


def _best_gradient() -> float:
    rows = _read_csv(STATISTICS_ROOT / GRADIENT_STUDY_KEY / "aggregate.csv")
    if len(rows) != len(GRADIENTS_G_PER_CM):
        raise ValueError("complete gradient sweep is required before the diameter sweep")
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["loading_rate_mean_atoms_per_s"]),
            float(row["gradient_g_per_cm"]),
        ),
    )
    return float(ranked[0]["gradient_g_per_cm"])


def _best_s0() -> float:
    rows = _read_csv(STATISTICS_ROOT / established.RAW_STUDY_KEY / "aggregate.csv")
    if len(rows) != len(established.RAW_SATURATION_VALUES):
        raise ValueError("complete corrected raw-saturation sweep is required before detuning")
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["loading_rate_mean_atoms_per_s"]),
            float(row["s0"]),
        ),
    )
    return float(ranked[0]["s0"])


def _load_plot_rows(study_key: str) -> list[dict[str, str]]:
    definition = SCAN_DEFINITIONS[study_key]
    if definition["source"] == "established":
        path = ESTABLISHED_STATISTICS / study_key / "aggregate.csv"
    else:
        path = STATISTICS_ROOT / study_key / "aggregate.csv"
    rows = _read_csv(path)
    if not rows:
        raise ValueError(f"no completed rows for {study_key}")
    return rows


def plot_study(study_key: str) -> Path:
    definition = SCAN_DEFINITIONS[study_key]
    rows = _load_plot_rows(study_key)
    xfield = str(definition["xfield"])
    rows.sort(key=lambda row: float(row[xfield]))
    x = np.asarray([float(row[xfield]) for row in rows])
    mean = np.asarray([float(row["loading_rate_mean_atoms_per_s"]) for row in rows]) / 1e6
    lower = np.asarray([float(row["loading_rate_t95_lower_atoms_per_s"]) for row in rows]) / 1e6
    upper = np.asarray([float(row["loading_rate_t95_upper_atoms_per_s"]) for row in rows]) / 1e6

    figure, axis = plt.subplots(figsize=(10.8, 7.4))
    figure.patch.set_facecolor("#fbfaf6")
    axis.set_facecolor("#fbfaf6")
    axis.fill_between(
        x,
        lower,
        upper,
        color="#2a9d8f",
        alpha=0.22,
        linewidth=0,
        label="95% Student-t interval across direction discs",
    )
    axis.plot(
        x,
        mean,
        "o-",
        color="#0f766e",
        markerfacecolor="#fbfaf6",
        markeredgewidth=1.5,
        linewidth=2.0,
        markersize=5.8,
        label="Mean loading rate",
    )
    axis.set_title(str(definition["title"]), fontsize=15, pad=13)
    fixed_text = str(definition["fixed"])
    if study_key == established.DETUNING_STUDY_KEY:
        fixed_text = (
            f"s₀ = {float(rows[0]['s0']):g}; "
            f"P = {float(rows[0]['cooling_power_mw_per_beam']):.3f} mW/beam; "
            "Gz = 10 G/cm; D = 12.7 mm"
        )
    if study_key != established.DETUNING_STUDY_KEY:
        axis.text(
            0.5,
            1.005,
            fixed_text,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9.4,
            color="#334155",
        )
    axis.set_xlabel(str(definition["xlabel"]))
    if study_key == established.DETUNING_STUDY_KEY:
        linewidth_mhz = default_simple_mot_config().linewidth_hz / 1.0e6
        secondary = axis.secondary_xaxis(
            "top",
            functions=(
                lambda normalized: normalized * linewidth_mhz,
                lambda detuning_mhz: detuning_mhz / linewidth_mhz,
            ),
        )
        secondary.set_xlabel(
            rf"Ordinary-frequency detuning $\Delta_\nu=\Delta_\omega/(2\pi)$ "
            rf"[MHz]  ($\Gamma/(2\pi)={linewidth_mhz:g}$ MHz)"
        )
    axis.set_ylabel(r"Loading rate [$10^6$ atoms/s]")
    axis.grid(True, alpha=0.23)
    axis.legend(frameon=False, loc="best")
    caption = (
        "Deterministic effective two-level Rb-87 D2 mean force; gravity on; recoil diffusion off.\n"
        "25 full-sphere direction discs × 25 uniform-area points/disc; 15 mm disc radius; "
        "95% t interval, df = 24; no 4π multiplier."
    )
    if study_key == established.DETUNING_STUDY_KEY:
        linewidth_mhz = default_simple_mot_config().linewidth_hz / 1.0e6
        caption = "Fixed parameters: " + fixed_text + "\n" + caption
        caption += (
            f"\nConversion: Δω/Γ = Δν/(Γ/2π); one normalized unit is "
            f"{linewidth_mhz:g} MHz, with no extra 2π factor."
        )
    if study_key == DIAMETER_STUDY_KEY:
        gradient = float(rows[0]["gradient_g_per_cm"])
        axis.text(
            0.5,
            0.955,
            f"Selected gradient from preceding scan: Gz = {gradient:g} G/cm",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=9.4,
            color="#334155",
        )
    figure.text(0.5, 0.025, caption, ha="center", va="bottom", fontsize=8.8, color="#475569")
    figure.subplots_adjust(
        left=0.12,
        right=0.975,
        top=0.82 if study_key == established.DETUNING_STUDY_KEY else 0.87,
        bottom=0.23 if study_key == established.DETUNING_STUDY_KEY else 0.18,
    )
    output = FIGURES_ROOT / str(definition["filename"])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output


def _manifest() -> dict[str, object]:
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema": "pmot.mot_simple.loading-parameter-scans.v1",
        "campaign_name": CAMPAIGN_NAME,
        "created_utc": _utc_now(),
        "model": "deterministic effective two-level Rb-87 D2 MOT mean force",
        "reused_established_campaign": ESTABLISHED_NAME,
        "archived_studies_not_used_in_current_plots": list(established.STUDY_ORDER),
        "archived_exclusion_reason": (
            "September 2026 trajectories predate the authoritative correction from "
            "linewidth_hz/2 to pi*linewidth_hz in the scattering prefactor"
        ),
        "new_studies": list(SCAN_DEFINITIONS),
        "gradient_values_g_per_cm": list(GRADIENTS_G_PER_CM),
        "diameter_values_mm": list(DIAMETERS_MM),
        "new_scan_fixed_s0": OPTIMAL_S0,
        "new_scan_fixed_detuning_delta_over_gamma": OPTIMAL_DETUNING_N,
        "gradient_scan_fixed_diameter_mm": REFERENCE_DIAMETER_MM,
        "diameter_power_rule": "constant Gaussian-center s0; P(D)=P(12.7 mm)*(D/12.7 mm)^2",
        "diameter_gradient_selection": "largest sampled mean loading rate; lower gradient breaks exact ties",
        "sampling": asdict(established.default_search_config()),
        "uncertainty": "two-sided 95% Student-t interval across 25 direction-disc clusters (df=24)",
        "plot_uncertainty_rendering": "translucent fill_between band around sampled means",
        "runner_sha256": script_hash,
    }


def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def audit_campaign() -> dict[str, object]:
    """Independently verify clustered statistics and point completeness."""

    global _ACTIVE_DIAMETER_GRADIENT_G_PER_CM
    _ACTIVE_DIAMETER_GRADIENT_G_PER_CM = _best_gradient()
    checks: list[dict[str, object]] = []
    for study_key in SCAN_DEFINITIONS:
        points = _points(study_key)
        aggregate_path = STATISTICS_ROOT / study_key / "aggregate.csv"
        aggregate = _read_csv(aggregate_path)
        if len(aggregate) != len(points):
            raise ValueError(
                f"{study_key} has {len(aggregate)} of {len(points)} aggregate rows"
            )
        aggregate_by_value = {float(row["scan_value"]): row for row in aggregate}
        for point in points:
            row = aggregate_by_value[point.scan_value]
            paths = _point_paths(point)
            metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
            loading = json.loads(paths.loading_json.read_text(encoding="utf-8"))
            if metadata.get("status") != "completed":
                raise ValueError(f"{point.slug} metadata is not completed")
            file_counts = {
                "samples": _csv_data_row_count(paths.final_samples_csv),
                "endpoint_audits": _csv_data_row_count(
                    paths.statistics / "capture_endpoint_audit.csv"
                ),
                "disc_loading_rates": _csv_data_row_count(paths.loading_by_disc_csv),
            }
            if file_counts != {
                "samples": 625,
                "endpoint_audits": 625,
                "disc_loading_rates": 25,
            }:
                raise ValueError(f"{point.slug} file counts are incomplete: {file_counts}")
            disc_rows = _read_csv(paths.loading_by_disc_csv)
            values = np.asarray(
                [float(item["loading_rate_atoms_per_s"]) for item in disc_rows]
            )
            mean = float(np.mean(values))
            sample_std = float(np.std(values, ddof=1))
            sem = sample_std / np.sqrt(values.size)
            critical = float(student_t.ppf(0.975, df=values.size - 1))
            lower = mean - critical * sem
            upper = mean + critical * sem
            recomputed = {
                "loading_rate_mean_atoms_per_s": mean,
                "loading_rate_sample_std_atoms_per_s": sample_std,
                "loading_rate_disc_cluster_sem_atoms_per_s": sem,
                "student_t_critical_95": critical,
                "loading_rate_t95_lower_atoms_per_s": lower,
                "loading_rate_t95_upper_atoms_per_s": upper,
            }
            for field, value in recomputed.items():
                if not np.isclose(value, float(loading[field]), rtol=2e-13, atol=1e-7):
                    raise ValueError(f"{point.slug} failed clustered-statistic audit: {field}")
                if field in row and not np.isclose(
                    value, float(row[field]), rtol=2e-13, atol=1e-7
                ):
                    raise ValueError(f"{point.slug} aggregate mismatch: {field}")
            if int(row["unresolved_timeout_count"]) != 0:
                raise ValueError(f"{point.slug} contains unresolved timeouts")
            checks.append(
                {
                    "study_key": study_key,
                    "point": point.scan_value,
                    "file_counts": file_counts,
                    "mean_atoms_per_s": mean,
                    "t95_lower_atoms_per_s": lower,
                    "t95_upper_atoms_per_s": upper,
                    "velocity_resolved_override_count": int(
                        row["velocity_resolved_override_count"]
                    ),
                    "indeterminate_zero_flux_ray_count": int(
                        row["indeterminate_zero_flux_ray_count"]
                    ),
                    "status": "passed",
                }
            )

    plot_paths = [FIGURES_ROOT / str(item["filename"]) for item in SCAN_DEFINITIONS.values()]
    if any(not path.is_file() or path.stat().st_size == 0 for path in plot_paths):
        raise ValueError("one or more final plot files are missing or empty")
    result = {
        "schema": "pmot.mot_simple.loading-parameter-scans.audit.v1",
        "audited_utc": _utc_now(),
        "status": "passed",
        "selected_diameter_gradient_g_per_cm": _ACTIVE_DIAMETER_GRADIENT_G_PER_CM,
        "new_point_count": len(checks),
        "new_ray_count": 625 * len(checks),
        "established_point_count_used": 0,
        "plot_count": len(plot_paths),
        "checks": checks,
    }
    _write_json(STATISTICS_ROOT / "final_audit.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("raw", "effective", "detuning", "gradient", "diameter", "plots", "audit", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--workers", type=int, default=WORKER_LIMIT)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_DIAMETER_GRADIENT_G_PER_CM
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= WORKER_LIMIT:
        raise ValueError(f"workers must be in 1..{WORKER_LIMIT}")
    STATISTICS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(STATISTICS_ROOT / "campaign_manifest.json", _manifest())
    if args.stage in {"raw", "all"}:
        run_study(established.RAW_STUDY_KEY, workers=args.workers, resume=args.resume)
    if args.stage in {"effective", "all"}:
        run_study(established.EFFECTIVE_STUDY_KEY, workers=args.workers, resume=args.resume)
    if args.stage in {"detuning", "all"}:
        run_study(established.DETUNING_STUDY_KEY, workers=args.workers, resume=args.resume)
    if args.stage in {"gradient", "all"}:
        run_study(GRADIENT_STUDY_KEY, workers=args.workers, resume=args.resume)
    if args.stage in {"diameter", "all"}:
        _ACTIVE_DIAMETER_GRADIENT_G_PER_CM = _best_gradient()
        run_study(DIAMETER_STUDY_KEY, workers=args.workers, resume=args.resume)
    if args.stage in {"plots", "all"}:
        outputs = {key: str(plot_study(key)) for key in SCAN_DEFINITIONS}
        _write_json(
            STATISTICS_ROOT / "plot_manifest.json",
            {
                "created_utc": _utc_now(),
                "plots": outputs,
                "statistics_root": str(STATISTICS_ROOT.resolve()),
            },
        )
        print(json.dumps(outputs, indent=2), flush=True)
    if args.stage in {"audit", "all"}:
        print(json.dumps(audit_campaign(), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
