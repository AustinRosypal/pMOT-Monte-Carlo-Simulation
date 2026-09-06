"""Checkpointed deterministic two-level MOT force sweep versus detuning.

The sweep is intentionally isolated from the multilevel implementation.  It
uses the authoritative effective two-level force in :mod:`simulation`, six
27 mW cooling components, the default 12.7 mm beam diameter, and the default
10 G/cm anti-Helmholtz quadrupole.  Frequencies remain in ordinary Hz, as
required by ``mot_simple``.  Gravity is excluded because the reported
quantities are derivatives and extrema of radiation pressure itself.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..configuration import MOTApparatusConfig, default_mot_apparatus_config
from ..magnetic_fields import default_anti_helmholtz_config
from ..state import AtomState
from .configuration import SimpleMOTConfig, default_simple_mot_config, simple_mot_paths
from .simulation import SimpleMOTBeam, build_simple_mot_beams, mean_force_n


AXIS_LABELS = "xyz"
COOLING_POWER_W_PER_BEAM = 27.0e-3
COOLING_COMPONENT_COUNT = 6
EVALUATION_COUNT = 1
SCHEMA_VERSION = 1

DETUNING_N_VALUES: tuple[float, ...] = tuple(
    round(-0.5 - 0.05 * index, 12) for index in range(111)
)
"""Detunings from ``n=-0.5`` through ``n=-6.0`` in steps of ``-0.05``."""


@dataclass(frozen=True, slots=True)
class SimpleForceSweepNumerics:
    """Coarse resolutions and convergence thresholds for the force sweep.

    Reported restoring slopes use half ``position_step_m`` (0.05 mm by
    default), while coarse slopes use the configured 0.1 mm span.  Reported
    turnaround speeds use half ``velocity_step_m_per_s`` (0.125 m/s by
    default), while coarse speeds use the configured 0.25 m/s grid.
    """

    position_step_m: float = 0.1e-3
    velocity_extent_m_per_s: float = 50.0
    velocity_step_m_per_s: float = 0.25
    restoring_relative_tolerance: float = 0.02
    turnaround_absolute_tolerance_m_per_s: float = 0.05

    def validate(self) -> None:
        values = np.asarray(
            (
                self.position_step_m,
                self.velocity_extent_m_per_s,
                self.velocity_step_m_per_s,
                self.restoring_relative_tolerance,
                self.turnaround_absolute_tolerance_m_per_s,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("all force-sweep numerical controls must be finite and positive")
        intervals = self.velocity_extent_m_per_s / self.velocity_step_m_per_s
        if not np.isclose(intervals, round(intervals), rtol=0.0, atol=1.0e-12):
            raise ValueError(
                "velocity_extent_m_per_s must be an integer multiple of velocity_step_m_per_s"
            )


CSV_FIELDNAMES = (
    "point_index",
    "deterministic_evaluation_count",
    "detuning_n",
    "detuning_hz",
    "detuning_mhz",
    "linewidth_hz",
    "cooling_power_w_per_beam",
    "cooling_beam_diameter_m",
    *(
        field
        for axis in AXIS_LABELS
        for field in (
            f"restoring_slope_{axis}_n_per_m",
            f"restoring_slope_{axis}_coarse_n_per_m",
            f"restoring_slope_{axis}_relative_change",
            f"restoring_slope_{axis}_numerical_uncertainty_n_per_m",
            f"restoring_slope_{axis}_converged",
            f"turnaround_velocity_{axis}_m_per_s",
            f"turnaround_velocity_{axis}_coarse_m_per_s",
            f"turnaround_velocity_{axis}_absolute_change_m_per_s",
            f"turnaround_velocity_{axis}_numerical_uncertainty_m_per_s",
            f"turnaround_force_{axis}_n",
            f"turnaround_{axis}_interior",
            f"turnaround_{axis}_converged",
        )
    ),
    "all_converged",
    "point_wall_time_s",
)

AxisProgressCallback = Callable[[str], None]


def detuning_n_grid() -> np.ndarray:
    """Return the exact 111-point detuning grid requested for comparison."""

    return np.asarray(DETUNING_N_VALUES, dtype=float).copy()


def _validate_detuning_values(values: Sequence[float]) -> tuple[float, ...]:
    detunings = tuple(float(value) for value in values)
    if not detunings:
        raise ValueError("at least one detuning is required")
    array = np.asarray(detunings, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array >= 0.0):
        raise ValueError("all detuning multipliers must be finite and negative")
    if len(set(detunings)) != len(detunings):
        raise ValueError("detuning multipliers must be unique")
    return detunings


def build_simple_force_sweep_configuration(
    detuning_n: float,
) -> tuple[SimpleMOTConfig, MOTApparatusConfig, list[SimpleMOTBeam]]:
    """Build synchronized ordinary-Hz two-level configuration and beams."""

    if not np.isfinite(detuning_n) or detuning_n >= 0.0:
        raise ValueError("detuning_n must be a finite negative number")
    base = default_simple_mot_config()
    detuning_hz = float(detuning_n * base.linewidth_hz)
    simple = replace(
        base,
        cooling_detuning_hz=detuning_hz,
        include_gravity=False,
    )
    apparatus = default_mot_apparatus_config()
    apparatus = replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            detuning_hz=detuning_hz,
            power_w_per_beam=COOLING_POWER_W_PER_BEAM,
        ),
    )
    beams = build_simple_mot_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    if len(beams) != COOLING_COMPONENT_COUNT:
        raise RuntimeError("two-level force sweep requires exactly six cooling beams")
    if not all(
        np.isclose(beam.detuning_hz, detuning_hz, rtol=0.0, atol=1.0e-9)
        for beam in beams
    ):
        raise RuntimeError("simple configuration, apparatus, and beam detunings differ")
    if not all(
        np.isclose(
            beam.intensity_beam.power_w,
            COOLING_POWER_W_PER_BEAM,
            rtol=0.0,
            atol=1.0e-15,
        )
        for beam in beams
    ):
        raise RuntimeError("built cooling-beam powers do not match 27 mW")
    if not all(
        np.isclose(
            beam.intensity_beam.beam_radius_m,
            0.5 * apparatus.cooling.beam_diameter_m,
            rtol=0.0,
            atol=1.0e-15,
        )
        for beam in beams
    ):
        raise RuntimeError("built cooling-beam diameters do not match the apparatus")
    return simple, apparatus, beams


def _axis_component_force_n(
    beams: list[SimpleMOTBeam],
    coil_config,
    simple_config: SimpleMOTConfig,
    axis_index: int,
    *,
    coordinate_m: float = 0.0,
    velocity_m_per_s: float = 0.0,
) -> float:
    position = np.zeros(3, dtype=float)
    velocity = np.zeros(3, dtype=float)
    position[axis_index] = coordinate_m
    velocity[axis_index] = velocity_m_per_s
    force, _, _ = mean_force_n(
        beams,
        AtomState(tuple(position), tuple(velocity)),
        coil_config,
        simple_config,
    )
    return float(force[axis_index])


def _central_restoring_slope_n_per_m(
    beams: list[SimpleMOTBeam],
    coil_config,
    simple_config: SimpleMOTConfig,
    axis_index: int,
    half_span_m: float,
) -> float:
    force_plus = _axis_component_force_n(
        beams,
        coil_config,
        simple_config,
        axis_index,
        coordinate_m=half_span_m,
    )
    force_minus = _axis_component_force_n(
        beams,
        coil_config,
        simple_config,
        axis_index,
        coordinate_m=-half_span_m,
    )
    return (force_plus - force_minus) / (2.0 * half_span_m)


def _parabolic_minimum(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, bool]:
    """Return the sampled minimum refined by a local three-point parabola."""

    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 3:
        raise ValueError("coordinates and values must be matching 1D arrays of length >= 3")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("turnaround samples must be finite and strictly ordered")
    minimum_index = int(np.argmin(y))
    if minimum_index in (0, x.size - 1):
        return float(x[minimum_index]), float(y[minimum_index]), False
    local_x = x[minimum_index - 1 : minimum_index + 2]
    local_y = y[minimum_index - 1 : minimum_index + 2]
    coefficients = np.polyfit(local_x, local_y, deg=2)
    curvature, linear, constant = coefficients
    if not np.all(np.isfinite(coefficients)) or curvature <= 0.0:
        return float(x[minimum_index]), float(y[minimum_index]), True
    vertex = float(np.clip(-linear / (2.0 * curvature), local_x[0], local_x[-1]))
    value = float(curvature * vertex**2 + linear * vertex + constant)
    return vertex, value, True


def evaluate_simple_force_detuning_point(
    point_index: int,
    detuning_n: float,
    numerics: SimpleForceSweepNumerics,
    *,
    coil_config=None,
    axis_progress_callback: AxisProgressCallback | None = None,
) -> dict[str, object]:
    """Evaluate restoring slopes and damping turnarounds at one detuning."""

    numerics.validate()
    simple, apparatus, beams = build_simple_force_sweep_configuration(detuning_n)
    coil = coil_config or default_anti_helmholtz_config()
    started = perf_counter()
    row: dict[str, object] = {
        "point_index": int(point_index),
        "deterministic_evaluation_count": EVALUATION_COUNT,
        "detuning_n": float(detuning_n),
        "detuning_hz": float(simple.cooling_detuning_hz),
        "detuning_mhz": float(simple.cooling_detuning_hz / 1.0e6),
        "linewidth_hz": float(simple.linewidth_hz),
        "cooling_power_w_per_beam": float(apparatus.cooling.power_w_per_beam),
        "cooling_beam_diameter_m": float(apparatus.cooling.beam_diameter_m),
    }

    coarse_intervals = int(
        round(numerics.velocity_extent_m_per_s / numerics.velocity_step_m_per_s)
    )
    fine_velocities = np.linspace(
        0.0,
        numerics.velocity_extent_m_per_s,
        2 * coarse_intervals + 1,
    )
    convergence_flags: list[bool] = []
    for axis_index, axis_label in enumerate(AXIS_LABELS):
        coarse_slope = _central_restoring_slope_n_per_m(
            beams,
            coil,
            simple,
            axis_index,
            numerics.position_step_m,
        )
        fine_slope = _central_restoring_slope_n_per_m(
            beams,
            coil,
            simple,
            axis_index,
            0.5 * numerics.position_step_m,
        )
        slope_uncertainty = abs(fine_slope - coarse_slope)
        slope_scale = max(abs(fine_slope), np.finfo(float).tiny)
        slope_change = slope_uncertainty / slope_scale
        slope_converged = bool(slope_change <= numerics.restoring_relative_tolerance)

        fine_forces = np.asarray(
            [
                _axis_component_force_n(
                    beams,
                    coil,
                    simple,
                    axis_index,
                    velocity_m_per_s=float(speed),
                )
                for speed in fine_velocities
            ],
            dtype=float,
        )
        coarse_velocity, _, coarse_interior = _parabolic_minimum(
            fine_velocities[::2],
            fine_forces[::2],
        )
        fine_velocity, fine_force, fine_interior = _parabolic_minimum(
            fine_velocities,
            fine_forces,
        )
        turnaround_change = abs(fine_velocity - coarse_velocity)
        turnaround_converged = bool(
            coarse_interior
            and fine_interior
            and turnaround_change <= numerics.turnaround_absolute_tolerance_m_per_s
        )

        row.update(
            {
                f"restoring_slope_{axis_label}_n_per_m": float(fine_slope),
                f"restoring_slope_{axis_label}_coarse_n_per_m": float(coarse_slope),
                f"restoring_slope_{axis_label}_relative_change": float(slope_change),
                f"restoring_slope_{axis_label}_numerical_uncertainty_n_per_m": float(
                    slope_uncertainty
                ),
                f"restoring_slope_{axis_label}_converged": slope_converged,
                f"turnaround_velocity_{axis_label}_m_per_s": float(fine_velocity),
                f"turnaround_velocity_{axis_label}_coarse_m_per_s": float(coarse_velocity),
                f"turnaround_velocity_{axis_label}_absolute_change_m_per_s": float(
                    turnaround_change
                ),
                f"turnaround_velocity_{axis_label}_numerical_uncertainty_m_per_s": float(
                    turnaround_change
                ),
                f"turnaround_force_{axis_label}_n": float(fine_force),
                f"turnaround_{axis_label}_interior": bool(fine_interior),
                f"turnaround_{axis_label}_converged": turnaround_converged,
            }
        )
        convergence_flags.extend((slope_converged, turnaround_converged))
        if axis_progress_callback is not None:
            axis_progress_callback(axis_label)

    row["all_converged"] = bool(all(convergence_flags))
    row["point_wall_time_s"] = float(perf_counter() - started)
    return row


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(0.05 * (attempt + 1))


def _write_csv_checkpoint(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["point_index"])):
            writer.writerow({field: row.get(field) for field in CSV_FIELDNAMES})
    _replace_with_retry(temporary, path)


def _read_csv_checkpoint(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _replace_with_retry(temporary, path)


def _rows_from_source(
    rows_or_csv: Sequence[Mapping[str, object]] | Path,
) -> list[Mapping[str, object]]:
    rows = _read_csv_checkpoint(rows_or_csv) if isinstance(rows_or_csv, Path) else list(rows_or_csv)
    if not rows:
        raise ValueError("at least one completed force row is required")
    return sorted(rows, key=lambda row: float(row["detuning_n"]))


_AXIS_STYLES = {
    "x": {"color": "#2563eb", "linestyle": "-", "marker": "o"},
    "y": {"color": "#d97706", "linestyle": "--", "marker": "s"},
    "z": {"color": "#059669", "linestyle": ":", "marker": "^"},
}


def _dense_display_indices(point_count: int, axis_index: int) -> np.ndarray:
    if point_count <= 40:
        return np.arange(point_count, dtype=int)
    return np.arange(axis_index, point_count, 4, dtype=int)


def _xy_overlap(values_x: np.ndarray, values_y: np.ndarray) -> bool:
    return bool(np.allclose(values_x, values_y, rtol=2.0e-10, atol=1.0e-30))


def _show_detuning_endpoints(axis, n_values: np.ndarray) -> None:
    lower = float(np.min(n_values))
    upper = float(np.max(n_values))
    span = upper - lower
    margin = 0.02 * span if span > 0.0 else 0.1
    axis.set_xlim(lower - margin, upper + margin)
    ticks = np.asarray(axis.get_xticks(), dtype=float)
    ticks = ticks[(ticks > lower) & (ticks < upper)]
    axis.set_xticks(np.unique(np.concatenate(([lower], ticks, [upper]))))


def _style_summary_axis(axis) -> None:
    axis.grid(True, which="major", alpha=0.25, linewidth=0.8)
    axis.grid(True, which="minor", alpha=0.10, linewidth=0.5)
    axis.minorticks_on()
    axis.tick_params(direction="in", top=True, right=True)


def _force_plot_caption() -> str:
    return (
        "Effective two-level deterministic radiation pressure; six 27 mW cooling beams, "
        "12.7 mm diameter, 10 G/cm axial quadrupole. Gravity excluded. Whiskers are "
        "|fine - coarse| numerical-resolution estimates (not statistical error bars)."
    )


def plot_simple_restoring_slopes_vs_detuning(
    rows_or_csv: Sequence[Mapping[str, object]] | Path,
    output_path: Path,
) -> Path:
    """Plot signed central restoring-force slopes for all Cartesian axes."""

    rows = _rows_from_source(rows_or_csv)
    n_values = np.asarray([float(row["detuning_n"]) for row in rows])
    values = {
        axis: np.asarray([float(row[f"restoring_slope_{axis}_n_per_m"]) for row in rows])
        for axis in AXIS_LABELS
    }
    uncertainties = {
        axis: np.asarray(
            [
                float(row.get(f"restoring_slope_{axis}_numerical_uncertainty_n_per_m", 0.0))
                for row in rows
            ]
        )
        for axis in AXIS_LABELS
    }
    figure, plot_axis = plt.subplots(figsize=(9.4, 7.4))
    figure.subplots_adjust(left=0.14, right=0.985, top=0.81, bottom=0.24)
    overlap = _xy_overlap(values["x"], values["y"])
    for axis_index, label in enumerate(AXIS_LABELS):
        display = _dense_display_indices(len(rows), axis_index)
        style = _AXIS_STYLES[label]
        plot_axis.errorbar(
            n_values[display],
            values[label][display],
            yerr=uncertainties[label][display],
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.7,
            zorder=3,
        )
        plot_axis.plot(
            n_values,
            values[label],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=(axis_index, 4) if len(rows) > 40 else None,
            markersize=4.2,
            linewidth=1.7,
            label=rf"${label}$ axis" + (" (coincident x/y pair)" if overlap and label in "xy" else ""),
            zorder=4 + axis_index,
        )
        unconverged = np.asarray(
            [not _bool_value(row[f"restoring_slope_{label}_converged"]) for row in rows]
        )
        if np.any(unconverged):
            plot_axis.scatter(
                n_values[unconverged],
                values[label][unconverged],
                marker="x",
                s=58,
                color="#dc2626",
                zorder=10,
            )
    if overlap:
        plot_axis.text(
            0.0,
            1.055,
            r"$x=y$ to numerical precision; distinct dashes and staggered markers expose both traces",
            transform=plot_axis.transAxes,
            ha="left",
            va="top",
            clip_on=False,
        )
    plot_axis.text(
        0.0,
        -0.21,
        _force_plot_caption(),
        transform=plot_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        clip_on=False,
    )
    plot_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    plot_axis.set(
        xlabel=r"detuning multiplier $n$ in $\Delta=n\Gamma$",
        ylabel=r"central restoring slope $\partial F_i/\partial x_i$ [N m$^{-1}$]",
    )
    figure.suptitle(
        "Two-level MOT restoring-force slope versus cooling detuning",
        fontsize=16,
        y=0.965,
    )
    _show_detuning_endpoints(plot_axis, n_values)
    _style_summary_axis(plot_axis)
    plot_axis.legend(loc="best", framealpha=0.96)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return destination


def plot_simple_damping_turnarounds_vs_detuning(
    rows_or_csv: Sequence[Mapping[str, object]] | Path,
    output_path: Path,
) -> Path:
    """Plot the positive-speed damping-force minimum for all axes."""

    rows = _rows_from_source(rows_or_csv)
    n_values = np.asarray([float(row["detuning_n"]) for row in rows])
    values = {
        axis: np.asarray([float(row[f"turnaround_velocity_{axis}_m_per_s"]) for row in rows])
        for axis in AXIS_LABELS
    }
    uncertainties = {
        axis: np.asarray(
            [
                float(row.get(f"turnaround_velocity_{axis}_numerical_uncertainty_m_per_s", 0.0))
                for row in rows
            ]
        )
        for axis in AXIS_LABELS
    }
    figure, plot_axis = plt.subplots(figsize=(9.4, 7.4))
    figure.subplots_adjust(left=0.125, right=0.985, top=0.78, bottom=0.27)
    xy_overlap = _xy_overlap(values["x"], values["y"])
    all_overlap = xy_overlap and _xy_overlap(values["x"], values["z"])
    for axis_index, label in enumerate(AXIS_LABELS):
        display = _dense_display_indices(len(rows), axis_index)
        style = _AXIS_STYLES[label]
        plot_axis.errorbar(
            n_values[display],
            values[label][display],
            yerr=uncertainties[label][display],
            fmt="none",
            ecolor=style["color"],
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.7,
            zorder=3,
        )
        plot_axis.plot(
            n_values,
            values[label],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=(axis_index, 4) if len(rows) > 40 else None,
            markersize=4.2,
            linewidth=1.7,
            label=rf"${label}$ axis" + (" (coincident x/y/z trio)" if all_overlap else ""),
            zorder=4 + axis_index,
        )
        unconverged = np.asarray(
            [not _bool_value(row[f"turnaround_{label}_converged"]) for row in rows]
        )
        if np.any(unconverged):
            plot_axis.scatter(
                n_values[unconverged],
                values[label][unconverged],
                marker="x",
                s=58,
                color="#dc2626",
                zorder=10,
            )
    overlap_note = (
        r"$x=y=z$ to numerical precision; distinct dashes and staggered markers expose all traces"
        if all_overlap
        else r"$x=y$ to numerical precision; distinct dashes and staggered markers expose both traces"
    )
    if xy_overlap:
        plot_axis.text(
            0.0,
            1.07,
            overlap_note,
            transform=plot_axis.transAxes,
            ha="left",
            va="top",
            clip_on=False,
        )
    plot_axis.text(
        0.0,
        1.20,
        "Turnaround: positive speed where $F_i(v_i)$ is most negative\n"
        "(strongest force opposing +motion; local parabolic interpolation, not a zero crossing)",
        transform=plot_axis.transAxes,
        ha="left",
        va="top",
        clip_on=False,
    )
    plot_axis.text(
        0.0,
        -0.28,
        _force_plot_caption(),
        transform=plot_axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        clip_on=False,
    )
    plot_axis.set(
        xlabel=r"detuning multiplier $n$ in $\Delta=n\Gamma$",
        ylabel=r"damping-turnaround speed [m s$^{-1}$]",
    )
    figure.suptitle(
        "Two-level MOT damping-force turnaround versus cooling detuning",
        fontsize=16,
        y=0.965,
    )
    _show_detuning_endpoints(plot_axis, n_values)
    _style_summary_axis(plot_axis)
    plot_axis.legend(loc="upper right", framealpha=0.96)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return destination


def _detuning_key(value: float) -> str:
    return f"{float(value):.17g}"


def _json_compatible(value: object) -> object:
    return json.loads(json.dumps(value))


def _resume_signature(
    detuning_values: Sequence[float],
    numerics: SimpleForceSweepNumerics,
) -> dict[str, object]:
    simple, apparatus, beams = build_simple_force_sweep_configuration(detuning_values[0])
    coil = default_anti_helmholtz_config()
    return {
        "schema_version": SCHEMA_VERSION,
        "detuning_n_values": [float(value) for value in detuning_values],
        "deterministic_evaluation_count_per_point": EVALUATION_COUNT,
        "frequency_unit": "ordinary Hz",
        "cooling_power_w_per_beam": COOLING_POWER_W_PER_BEAM,
        "cooling_component_count": len(beams),
        "repump_component_count": 0,
        "numerics": asdict(numerics),
        "simple_config": _json_compatible(asdict(simple)),
        "apparatus_config": _json_compatible(asdict(apparatus)),
        "coil_config": _json_compatible(asdict(coil)),
    }


def _metadata_payload(
    *,
    signature: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    status: str,
    resume: bool,
    resumed_point_count: int,
    created_utc: str,
    wall_time_s: float,
    csv_path: Path,
    restoring_plot_path: Path,
    turnaround_plot_path: Path,
) -> dict[str, object]:
    base = default_simple_mot_config()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "created_utc": created_utc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "resume_enabled": resume,
        "resumed_point_count": resumed_point_count,
        "completed_point_count": len(rows),
        "total_point_count": len(signature["detuning_n_values"]),
        "deterministic_evaluation_count_per_point": EVALUATION_COUNT,
        "total_deterministic_evaluations": len(rows) * EVALUATION_COUNT,
        "wall_time_s_current_invocation": wall_time_s,
        "resume_signature": dict(signature),
        "all_convergence_checks_passed": bool(
            rows and all(_bool_value(row["all_converged"]) for row in rows)
        ),
        "model": "deterministic effective two-level MOT mean radiation-pressure force",
        "frequency_unit": "ordinary Hz",
        "detuning_definition": "Delta = n Gamma; Delta and Gamma are ordinary frequencies in Hz",
        "restoring_slope_definition": (
            "signed dF_i/dx_i at the origin and zero velocity; a restoring slope is negative; "
            "radiation pressure only, gravity excluded"
        ),
        "turnaround_definition": (
            "positive v_i where F_i(v_i) is most negative (strongest force opposing positive "
            "motion), refined by a local three-point parabola; not a force zero crossing"
        ),
        "convergence_definition": (
            "reported slopes use +/-0.05 mm and are compared with +/-0.1 mm; reported "
            "turnarounds use a 0.125 m/s grid and are compared with a 0.25 m/s grid"
        ),
        "statistical_uncertainty_applicable": False,
        "numerical_uncertainty_definition": (
            "absolute fine-minus-coarse difference from one deterministic calculation"
        ),
        "cooling_power_w_per_beam": COOLING_POWER_W_PER_BEAM,
        "cooling_power_mw_per_beam": 1.0e3 * COOLING_POWER_W_PER_BEAM,
        "cooling_component_count": COOLING_COMPONENT_COUNT,
        "repumper_enabled": False,
        "repump_component_count": 0,
        "beam_diameter_m": float(
            signature["apparatus_config"]["cooling"]["beam_diameter_m"]
        ),
        "beam_diameter_mm": 1.0e3
        * float(signature["apparatus_config"]["cooling"]["beam_diameter_m"]),
        "quadrupole_axial_gradient_g_per_cm": 10.0,
        "gravity_included_in_force": False,
        "linewidth_hz": base.linewidth_hz,
        "limitations": [
            "The effective two-level model omits hyperfine/Zeeman populations, repumping, coherences, and sub-Doppler physics.",
            "Turnaround is the global force minimum within the configured positive-speed window.",
            "Rows that fail coarse/fine convergence remain checkpointed and are visibly flagged.",
        ],
        "outputs": {
            "force_sweep_csv": str(csv_path),
            "restoring_slope_plot": str(restoring_plot_path),
            "damping_turnaround_plot": str(turnaround_plot_path),
        },
    }


def run_simple_force_detuning_sweep(
    *,
    detuning_n_values: Sequence[float] | None = None,
    numerics: SimpleForceSweepNumerics | None = None,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """Run, checkpoint, resume, and plot the two-level force sweep."""

    values = _validate_detuning_values(
        DETUNING_N_VALUES if detuning_n_values is None else detuning_n_values
    )
    controls = numerics or SimpleForceSweepNumerics()
    controls.validate()
    paths = simple_mot_paths()
    output = output_directory or paths["outputs_statistics_simple_mot"] / "force_vs_detuning_27mW"
    figures = figure_directory or paths["outputs_figures_simple_mot"] / "force_vs_detuning_27mW"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    csv_path = output / "force_vs_detuning.csv"
    metadata_path = output / "force_vs_detuning_metadata.json"
    restoring_plot_path = figures / "restoring_slope_vs_detuning.png"
    turnaround_plot_path = figures / "damping_turnaround_vs_detuning.png"
    signature = _resume_signature(values, controls)

    created_utc = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    if resume and (csv_path.exists() or metadata_path.exists()):
        if not csv_path.exists() or not metadata_path.exists():
            raise RuntimeError("resume requires both force-sweep CSV and metadata checkpoints")
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior_metadata.get("resume_signature") != signature:
            raise ValueError("existing force-sweep checkpoint has incompatible parameters")
        created_utc = str(prior_metadata.get("created_utc", created_utc))
        rows = _read_csv_checkpoint(csv_path)

    allowed = {_detuning_key(value): index for index, value in enumerate(values)}
    completed_by_key: dict[str, dict[str, object]] = {}
    for row in rows:
        key = _detuning_key(float(row["detuning_n"]))
        if key not in allowed:
            raise ValueError(f"checkpoint contains unexpected detuning n={row['detuning_n']}")
        if int(row["point_index"]) != allowed[key]:
            raise ValueError("checkpoint point index does not match detuning")
        if key in completed_by_key:
            raise ValueError(f"checkpoint contains duplicate detuning n={row['detuning_n']}")
        completed_by_key[key] = row
    rows = list(completed_by_key.values())
    resumed_point_count = len(rows)
    remaining = [
        (index, value)
        for index, value in enumerate(values)
        if _detuning_key(value) not in completed_by_key
    ]

    coil = default_anti_helmholtz_config()
    started = perf_counter()
    completed_new_axes = 0
    total_new_axes = 3 * len(remaining)
    print(
        f"[simple-force] points={len(values)}; resumed={resumed_point_count}; "
        f"remaining={len(remaining)}; six cooling beams at 27 mW; no repumper",
        flush=True,
    )
    _write_csv_checkpoint(csv_path, rows)
    _atomic_write_json(
        metadata_path,
        _metadata_payload(
            signature=signature,
            rows=rows,
            status="running" if remaining else "completed",
            resume=resume,
            resumed_point_count=resumed_point_count,
            created_utc=created_utc,
            wall_time_s=0.0,
            csv_path=csv_path,
            restoring_plot_path=restoring_plot_path,
            turnaround_plot_path=turnaround_plot_path,
        ),
    )

    for point_index, detuning_n in remaining:
        print(
            f"[simple-force] point {point_index + 1}/{len(values)} start: "
            f"n={detuning_n:+.2f}",
            flush=True,
        )

        def report_axis(axis_label: str) -> None:
            nonlocal completed_new_axes
            completed_new_axes += 1
            elapsed = perf_counter() - started
            eta_s = (
                (total_new_axes - completed_new_axes) * elapsed / completed_new_axes
                if completed_new_axes
                else 0.0
            )
            print(
                f"[simple-force] point {point_index + 1}/{len(values)}; axis={axis_label}; "
                f"new axes={completed_new_axes}/{total_new_axes}; elapsed={elapsed:.1f}s; "
                f"ETA={eta_s:.1f}s",
                flush=True,
            )

        row = evaluate_simple_force_detuning_point(
            point_index,
            detuning_n,
            controls,
            coil_config=coil,
            axis_progress_callback=report_axis,
        )
        rows.append(row)
        rows.sort(key=lambda item: int(item["point_index"]))
        _write_csv_checkpoint(csv_path, rows)
        _atomic_write_json(
            metadata_path,
            _metadata_payload(
                signature=signature,
                rows=rows,
                status="running",
                resume=resume,
                resumed_point_count=resumed_point_count,
                created_utc=created_utc,
                wall_time_s=perf_counter() - started,
                csv_path=csv_path,
                restoring_plot_path=restoring_plot_path,
                turnaround_plot_path=turnaround_plot_path,
            ),
        )
        print(
            f"[simple-force] point {point_index + 1}/{len(values)} checkpointed; "
            f"all_converged={row['all_converged']}; wall={row['point_wall_time_s']:.2f}s",
            flush=True,
        )

    if rows:
        plot_simple_restoring_slopes_vs_detuning(rows, restoring_plot_path)
        plot_simple_damping_turnarounds_vs_detuning(rows, turnaround_plot_path)
    status = "completed" if len(rows) == len(values) else "incomplete"
    final_metadata = _metadata_payload(
        signature=signature,
        rows=rows,
        status=status,
        resume=resume,
        resumed_point_count=resumed_point_count,
        created_utc=created_utc,
        wall_time_s=perf_counter() - started,
        csv_path=csv_path,
        restoring_plot_path=restoring_plot_path,
        turnaround_plot_path=turnaround_plot_path,
    )
    _write_csv_checkpoint(csv_path, rows)
    _atomic_write_json(metadata_path, final_metadata)
    print(
        f"[simple-force] {status}: {len(rows)}/{len(values)} points; "
        f"all_converged={final_metadata['all_convergence_checks_passed']}; "
        f"wall={perf_counter() - started:.1f}s",
        flush=True,
    )
    return {
        "status": status,
        "completed_point_count": len(rows),
        "total_point_count": len(values),
        "rows": rows,
        "outputs": {
            **final_metadata["outputs"],
            "metadata_json": str(metadata_path),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic two-level MOT detuning-force sweep"
    )
    parser.add_argument("--position-step-mm", type=float, default=0.1)
    parser.add_argument("--velocity-extent", type=float, default=50.0)
    parser.add_argument("--velocity-step", type=float, default=0.25)
    parser.add_argument("--restoring-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--turnaround-absolute-tolerance", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_simple_force_detuning_sweep(
        numerics=SimpleForceSweepNumerics(
            position_step_m=1.0e-3 * args.position_step_mm,
            velocity_extent_m_per_s=args.velocity_extent,
            velocity_step_m_per_s=args.velocity_step,
            restoring_relative_tolerance=args.restoring_relative_tolerance,
            turnaround_absolute_tolerance_m_per_s=args.turnaround_absolute_tolerance,
        ),
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
        resume=args.resume,
    )
    print(json.dumps(result["outputs"], indent=2), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COOLING_COMPONENT_COUNT",
    "COOLING_POWER_W_PER_BEAM",
    "DETUNING_N_VALUES",
    "EVALUATION_COUNT",
    "SimpleForceSweepNumerics",
    "build_argument_parser",
    "build_simple_force_sweep_configuration",
    "detuning_n_grid",
    "evaluate_simple_force_detuning_point",
    "main",
    "plot_simple_damping_turnarounds_vs_detuning",
    "plot_simple_restoring_slopes_vs_detuning",
    "run_simple_force_detuning_sweep",
]
