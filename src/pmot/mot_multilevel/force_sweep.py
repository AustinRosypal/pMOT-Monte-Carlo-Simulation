"""Checkpointed multilevel restoring/damping-force sweep versus detuning.

This module uses the quasi-steady population-rate observable from
``rate_equations.py``.  It does not use the legacy event-averaged force or the
effective two-level model.  Gravity is excluded because the reported values
are derivatives and extrema of radiation pressure itself.
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

from ..configuration import PMOTSimulationConfig, default_simulation_config
from ..fields import MOTBeam
from ..mot.magnetic_fields import default_anti_helmholtz_config
from .configuration import (
    MultilevelMOTConfig,
    default_multilevel_mot_config,
    multilevel_mot_paths,
)
from .rate_equations import (
    RateEquationModel,
    build_rate_equation_model,
    rate_equation_observable,
)
from .simulation import build_multilevel_mot_beams


DETUNING_N_VALUES: tuple[float, ...] = tuple(
    float(value) for value in np.linspace(-0.5, -10.0, 25)
)
"""Twenty-five uniformly spaced detunings, executed from weak to strong red."""

AXIS_LABELS = "xyz"
EVALUATION_COUNT = 1
"""One deterministic rate-equation force calculation per detuning point."""

# Kept as a compatibility alias for callers of the initial sweep draft.  This
# is an evaluation count, not a stochastic replicate count.
REPLICATE_COUNT = EVALUATION_COUNT

SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class ForceSweepNumerics:
    """Spatial/velocity resolutions and explicit convergence thresholds.

    ``position_step_m`` and ``velocity_step_m_per_s`` are the coarse
    resolutions.  Reported values use half those steps; the coarse/fine change
    is saved alongside every result.
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
            raise ValueError("velocity_extent_m_per_s must be an integer multiple of velocity_step_m_per_s")


CSV_FIELDNAMES = (
    "point_index",
    "deterministic_evaluation_count",
    "detuning_n",
    "detuning_rad_per_s",
    "detuning_hz",
    "detuning_mhz",
    "linewidth_rad_per_s",
    "model_state_count",
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
    """Return 25 values from ``n=-0.5`` through ``n=-10`` inclusive.

    The sweep is executed from weak to strong red detuning.  Plot helpers sort
    the completed points into increasing numerical order on the horizontal
    axis (``-10`` through ``-0.5``).
    """

    return np.asarray(DETUNING_N_VALUES, dtype=float).copy()


def _validate_detuning_values(values: Sequence[float]) -> tuple[float, ...]:
    detunings = tuple(float(value) for value in values)
    if not detunings:
        raise ValueError("at least one detuning is required")
    array = np.asarray(detunings, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array >= 0.0):
        raise ValueError("all detuning multipliers must be finite and negative (red detuning)")
    if len(set(detunings)) != len(detunings):
        raise ValueError("detuning multipliers must be unique")
    return detunings


def build_force_sweep_configuration(
    detuning_n: float,
) -> tuple[MultilevelMOTConfig, PMOTSimulationConfig, list[MOTBeam]]:
    """Build solver, apparatus, and beam metadata with one consistent detuning."""

    if not np.isfinite(detuning_n) or detuning_n >= 0.0:
        raise ValueError("detuning_n must be a finite negative number for red detuning")
    base = default_multilevel_mot_config()
    detuning_rad_per_s = float(detuning_n * base.natural_linewidth_rad_per_s)
    config = replace(
        base,
        cooling_detuning_rad_per_s=detuning_rad_per_s,
        repumper_enabled=True,
    )
    apparatus = default_simulation_config()
    apparatus = replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            detuning_hz=detuning_rad_per_s / (2.0 * np.pi),
        ),
    )
    beams = build_multilevel_mot_beams(apparatus_config=apparatus, config=config)
    return config, apparatus, beams


def _axis_component_force_n(
    model: RateEquationModel,
    beams: list[MOTBeam],
    coil_config,
    config: MultilevelMOTConfig,
    axis_index: int,
    *,
    coordinate_m: float = 0.0,
    velocity_m_per_s: float = 0.0,
) -> float:
    position = np.zeros(3, dtype=float)
    velocity = np.zeros(3, dtype=float)
    position[axis_index] = coordinate_m
    velocity[axis_index] = velocity_m_per_s
    return float(
        rate_equation_observable(
            model,
            beams,
            tuple(position),
            tuple(velocity),
            coil_config,
            config,
        ).force_n[axis_index]
    )


def _central_restoring_slope_n_per_m(
    model: RateEquationModel,
    beams: list[MOTBeam],
    coil_config,
    config: MultilevelMOTConfig,
    axis_index: int,
    half_span_m: float,
) -> float:
    force_plus = _axis_component_force_n(
        model,
        beams,
        coil_config,
        config,
        axis_index,
        coordinate_m=half_span_m,
    )
    force_minus = _axis_component_force_n(
        model,
        beams,
        coil_config,
        config,
        axis_index,
        coordinate_m=-half_span_m,
    )
    return (force_plus - force_minus) / (2.0 * half_span_m)


def _parabolic_minimum(
    coordinates: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, bool]:
    """Return a sampled minimum refined by its three-point local parabola.

    The boolean is true only when the selected minimum is interior to the
    sampled interval.  Boundary minima are retained but fail convergence so an
    insufficient velocity range cannot silently become a turnaround result.
    """

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
    if curvature <= 0.0 or not np.all(np.isfinite(coefficients)):
        return float(x[minimum_index]), float(y[minimum_index]), True
    vertex = float(-linear / (2.0 * curvature))
    vertex = float(np.clip(vertex, local_x[0], local_x[-1]))
    value = float(curvature * vertex**2 + linear * vertex + constant)
    return vertex, value, True


def _evaluate_force_detuning_once(
    point_index: int,
    detuning_n: float,
    numerics: ForceSweepNumerics,
    *,
    model: RateEquationModel | None = None,
    coil_config=None,
    axis_progress_callback: AxisProgressCallback | None = None,
) -> dict[str, object]:
    """Compute one deterministic restoring/damping evaluation at one detuning.

    The signed restoring slope is ``dF_i/dx_i`` at the origin and zero
    velocity; a stable restoring force therefore has a negative slope.

    The positive damping turnaround is the ``v_i > 0`` location where
    ``F_i(v_i)`` is most negative, i.e. the strongest force opposing positive
    motion.  This is a force extremum, not a zero crossing.  A three-point
    parabolic interpolation refines the discrete minimum.
    """

    numerics.validate()
    config, _, beams = build_force_sweep_configuration(detuning_n)
    rate_model = model or build_rate_equation_model(config.natural_linewidth_rad_per_s)
    coil = coil_config or default_anti_helmholtz_config()
    started = perf_counter()
    gamma = config.natural_linewidth_rad_per_s
    row: dict[str, object] = {
        "point_index": int(point_index),
        "detuning_n": float(detuning_n),
        "detuning_rad_per_s": float(config.cooling_detuning_rad_per_s),
        "detuning_hz": float(config.cooling_detuning_rad_per_s / (2.0 * np.pi)),
        "detuning_mhz": float(config.cooling_detuning_rad_per_s / (2.0 * np.pi * 1.0e6)),
        "linewidth_rad_per_s": float(gamma),
        "model_state_count": int(rate_model.state_count),
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
            rate_model,
            beams,
            coil,
            config,
            axis_index,
            numerics.position_step_m,
        )
        fine_slope = _central_restoring_slope_n_per_m(
            rate_model,
            beams,
            coil,
            config,
            axis_index,
            0.5 * numerics.position_step_m,
        )
        slope_scale = max(abs(fine_slope), np.finfo(float).tiny)
        slope_change = abs(fine_slope - coarse_slope) / slope_scale
        slope_converged = bool(
            slope_change <= numerics.restoring_relative_tolerance
        )

        fine_forces = np.asarray(
            [
                _axis_component_force_n(
                    rate_model,
                    beams,
                    coil,
                    config,
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
            and turnaround_change
            <= numerics.turnaround_absolute_tolerance_m_per_s
        )

        row.update(
            {
                f"restoring_slope_{axis_label}_n_per_m": float(fine_slope),
                f"restoring_slope_{axis_label}_coarse_n_per_m": float(coarse_slope),
                f"restoring_slope_{axis_label}_relative_change": float(slope_change),
                f"restoring_slope_{axis_label}_converged": slope_converged,
                f"turnaround_velocity_{axis_label}_m_per_s": float(fine_velocity),
                f"turnaround_velocity_{axis_label}_coarse_m_per_s": float(coarse_velocity),
                f"turnaround_velocity_{axis_label}_absolute_change_m_per_s": float(
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


def evaluate_force_detuning_point(
    point_index: int,
    detuning_n: float,
    numerics: ForceSweepNumerics,
    *,
    model: RateEquationModel | None = None,
    coil_config=None,
    axis_progress_callback: AxisProgressCallback | None = None,
) -> dict[str, object]:
    """Perform one deterministic force calculation at one detuning.

    Statistical error bars do not apply to this deterministic observable.
    Fine/coarse spatial and velocity-grid differences are retained as honest
    numerical-resolution uncertainty estimates.
    """

    row = _evaluate_force_detuning_once(
        point_index,
        detuning_n,
        numerics,
        model=model,
        coil_config=coil_config,
        axis_progress_callback=axis_progress_callback,
    )
    row["deterministic_evaluation_count"] = EVALUATION_COUNT
    for axis_label in AXIS_LABELS:
        row[f"restoring_slope_{axis_label}_numerical_uncertainty_n_per_m"] = abs(
            float(row[f"restoring_slope_{axis_label}_n_per_m"])
            - float(row[f"restoring_slope_{axis_label}_coarse_n_per_m"])
        )
        row[
            f"turnaround_velocity_{axis_label}_numerical_uncertainty_m_per_s"
        ] = abs(
            float(row[f"turnaround_velocity_{axis_label}_m_per_s"])
            - float(row[f"turnaround_velocity_{axis_label}_coarse_m_per_s"])
        )
    return row


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Replace a checkpoint despite short-lived Windows sync-client locks."""

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
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _replace_with_retry(temporary, path)


def _rows_from_source(
    rows_or_csv: Sequence[Mapping[str, object]] | Path | str,
) -> list[Mapping[str, object]]:
    if isinstance(rows_or_csv, (str, Path)):
        rows: list[Mapping[str, object]] = _read_csv_checkpoint(Path(rows_or_csv))
    else:
        rows = list(rows_or_csv)
    if not rows:
        raise ValueError("at least one completed force-sweep row is required")
    return sorted(rows, key=lambda row: float(row["detuning_n"]))


_AXIS_STYLES = {
    "x": {
        "color": "#2563eb",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.8,
        "markersize": 7.0,
        "zorder": 4,
    },
    "y": {
        "color": "#ea580c",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.7,
        "markersize": 4.5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.3,
        "zorder": 6,
    },
    "z": {
        "color": "#15803d",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.0,
        "markersize": 6.0,
        "zorder": 5,
    },
}


def _style_summary_axis(axis) -> None:
    axis.axhline(0.0, color="#111827", linewidth=1.1, zorder=2)
    axis.grid(color="#94a3b8", alpha=0.28, linewidth=0.8, zorder=0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#111827")
        spine.set_linewidth(1.1)
        spine.set_zorder(10)
    axis.tick_params(axis="both", colors="#111827", width=1.0)


def _xy_overlap(values_x: np.ndarray, values_y: np.ndarray) -> bool:
    scale = max(float(np.max(np.abs(values_x))), float(np.max(np.abs(values_y))), 1.0e-300)
    return bool(np.max(np.abs(values_x - values_y)) <= 1.0e-10 * scale)


def plot_restoring_slopes_vs_detuning(
    rows_or_csv: Sequence[Mapping[str, object]] | Path | str,
    output_path: Path | str,
) -> Path:
    """Plot signed origin slopes with deliberately distinguishable axis traces."""

    rows = _rows_from_source(rows_or_csv)
    n_values = np.asarray([float(row["detuning_n"]) for row in rows])
    values_by_axis = {
        label: np.asarray(
            [float(row[f"restoring_slope_{label}_n_per_m"]) for row in rows]
        )
        for label in AXIS_LABELS
    }
    numerical_uncertainty_by_axis = {
        label: np.asarray(
            [
                float(
                    row.get(
                        f"restoring_slope_{label}_numerical_uncertainty_n_per_m",
                        0.0,
                    )
                )
                for row in rows
            ]
        )
        for label in AXIS_LABELS
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    overlap = _xy_overlap(values_by_axis["x"], values_by_axis["y"])
    all_axis_overlap = overlap and _xy_overlap(
        values_by_axis["x"], values_by_axis["z"]
    )
    for label in AXIS_LABELS:
        legend_label = rf"${label}$ axis"
        if all_axis_overlap:
            legend_label += " (coincident x/y/z trio)"
        elif overlap and label in "xy":
            legend_label += " (coincident x/y pair)"
        axis.errorbar(
            n_values,
            values_by_axis[label],
            yerr=numerical_uncertainty_by_axis[label],
            fmt="none",
            ecolor=_AXIS_STYLES[label]["color"],
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.65,
            zorder=3,
        )
        axis.plot(
            n_values,
            values_by_axis[label],
            label=legend_label,
            **_AXIS_STYLES[label],
        )
        unconverged = np.asarray(
            [
                not _bool_value(row[f"restoring_slope_{label}_converged"])
                for row in rows
            ]
        )
        if np.any(unconverged):
            axis.scatter(
                n_values[unconverged],
                values_by_axis[label][unconverged],
                marker="x",
                s=62,
                linewidths=1.6,
                color="#dc2626",
                zorder=9,
            )
    if all_axis_overlap:
        axis.text(
            0.02,
            0.84,
            r"$x=y=z$ to numerical precision; distinct dashes/markers expose all traces",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.92},
            zorder=12,
        )
    elif overlap:
        axis.text(
            0.02,
            0.84,
            r"$x=y$ to numerical precision; dashed squares expose the $y$ trace",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.92},
            zorder=12,
        )
    gamma = float(rows[0]["linewidth_rad_per_s"])
    axis.text(
        0.02,
        0.97,
        rf"$\Delta=n\Gamma$; $n<0$ is red; $\Gamma/(2\pi)={gamma/(2*np.pi*1e6):.2f}$ MHz",
        transform=axis.transAxes,
        ha="left",
        va="top",
        zorder=12,
    )
    axis.text(
        0.02,
        0.03,
        "One deterministic rate-equation calculation per detuning\n"
        r"Whiskers show $|$fine$-$coarse$|$ numerical-resolution difference; no statistical bars",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        zorder=12,
    )
    axis.set(
        xlabel=r"detuning multiplier $n$",
        ylabel=r"signed origin slope $\left.\partial F_i/\partial x_i\right|_0$ [N m$^{-1}$]",
        title="Multilevel MOT restoring-force slope versus cooling detuning",
    )
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    _style_summary_axis(axis)
    axis.legend(loc="best", framealpha=0.96)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190)
    plt.close(figure)
    return destination


def plot_damping_turnarounds_vs_detuning(
    rows_or_csv: Sequence[Mapping[str, object]] | Path | str,
    output_path: Path | str,
) -> Path:
    """Plot strongest-opposing-force velocities for all Cartesian axes."""

    rows = _rows_from_source(rows_or_csv)
    n_values = np.asarray([float(row["detuning_n"]) for row in rows])
    values_by_axis = {
        label: np.asarray(
            [float(row[f"turnaround_velocity_{label}_m_per_s"]) for row in rows]
        )
        for label in AXIS_LABELS
    }
    numerical_uncertainty_by_axis = {
        label: np.asarray(
            [
                float(
                    row.get(
                        f"turnaround_velocity_{label}_numerical_uncertainty_m_per_s",
                        0.0,
                    )
                )
                for row in rows
            ]
        )
        for label in AXIS_LABELS
    }
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    overlap = _xy_overlap(values_by_axis["x"], values_by_axis["y"])
    all_axis_overlap = overlap and _xy_overlap(
        values_by_axis["x"], values_by_axis["z"]
    )
    for label in AXIS_LABELS:
        legend_label = rf"${label}$ axis"
        if all_axis_overlap:
            legend_label += " (coincident x/y/z trio)"
        elif overlap and label in "xy":
            legend_label += " (coincident x/y pair)"
        axis.errorbar(
            n_values,
            values_by_axis[label],
            yerr=numerical_uncertainty_by_axis[label],
            fmt="none",
            ecolor=_AXIS_STYLES[label]["color"],
            elinewidth=0.9,
            capsize=2.0,
            alpha=0.65,
            zorder=3,
        )
        axis.plot(
            n_values,
            values_by_axis[label],
            label=legend_label,
            **_AXIS_STYLES[label],
        )
        unconverged = np.asarray(
            [
                not _bool_value(row[f"turnaround_{label}_converged"])
                for row in rows
            ]
        )
        if np.any(unconverged):
            axis.scatter(
                n_values[unconverged],
                values_by_axis[label][unconverged],
                marker="x",
                s=62,
                linewidths=1.6,
                color="#dc2626",
                zorder=9,
            )
    if all_axis_overlap:
        axis.text(
            0.02,
            0.84,
            r"$x=y=z$ to numerical precision; distinct dashes/markers expose all traces",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.92},
            zorder=12,
        )
    elif overlap:
        axis.text(
            0.02,
            0.84,
            r"$x=y$ to numerical precision; dashed squares expose the $y$ trace",
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#94a3b8", "alpha": 0.92},
            zorder=12,
        )
    axis.text(
        0.02,
        0.97,
        "Turnaround: positive speed where $F_i(v_i)$ is most negative\n"
        "(strongest force opposing +motion; local parabolic interpolation, not a zero crossing)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        zorder=12,
    )
    axis.text(
        0.02,
        0.075,
        "One deterministic rate-equation calculation per detuning\n"
        r"Whiskers show $|$fine$-$coarse$|$ numerical-resolution difference; no statistical bars",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        zorder=12,
    )
    axis.set(
        xlabel=r"detuning multiplier $n$ in $\Delta=n\Gamma$",
        ylabel=r"damping-turnaround speed [m s$^{-1}$]",
        title="Multilevel MOT damping-force turnaround versus cooling detuning",
    )
    _style_summary_axis(axis)
    axis.legend(loc="upper right", framealpha=0.96)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190)
    plt.close(figure)
    return destination


def _detuning_key(value: float) -> str:
    return f"{float(value):.17g}"


def _resume_signature(
    detuning_values: Sequence[float],
    numerics: ForceSweepNumerics,
) -> dict[str, object]:
    production_config = replace(
        default_multilevel_mot_config(), repumper_enabled=True
    )
    apparatus = default_simulation_config()
    coil = default_anti_helmholtz_config()
    model = build_rate_equation_model(production_config.natural_linewidth_rad_per_s)

    def json_compatible(value: object) -> object:
        return json.loads(json.dumps(value))

    return {
        "schema_version": SCHEMA_VERSION,
        "detuning_n_values": [float(value) for value in detuning_values],
        "deterministic_evaluation_count_per_point": EVALUATION_COUNT,
        "numerics": asdict(numerics),
        "multilevel_config": json_compatible(asdict(production_config)),
        "apparatus_config": json_compatible(asdict(apparatus)),
        "coil_config": json_compatible(asdict(coil)),
        "model_state_count": model.state_count,
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
    base = replace(default_multilevel_mot_config(), repumper_enabled=True)
    model = build_rate_equation_model(base.natural_linewidth_rad_per_s)
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
        "detuning_definition": "Delta = n Gamma; Gamma is angular frequency and negative n is red detuning",
        "restoring_slope_definition": (
            "signed dF_i/dx_i at r=0 and v=0; a stable restoring slope is negative; "
            "radiation pressure only, gravity excluded"
        ),
        "turnaround_definition": (
            "positive v_i where F_i(v_i) reaches its most negative value (strongest force "
            "opposing positive motion), refined by a local three-point parabola; this is "
            "an extremum, not a force zero crossing"
        ),
        "convergence_definition": (
            "reported restoring slope uses half the configured position step and is compared "
            "with the configured coarse step; reported turnaround uses half the configured "
            "velocity step and is compared with the coarse grid"
        ),
        "statistical_uncertainty_applicable": False,
        "numerical_uncertainty_definition": (
            "absolute fine-minus-coarse difference from the single deterministic calculation; "
            "plotted as numerical-resolution whiskers and not as statistical error"
        ),
        "solver": "quasi-steady multilevel population-rate force",
        "model_state_count": model.state_count,
        "cooling_only_specification_state_count": 23,
        "state_count_note": (
            "current model has 24 indexed states because the required repumper extension "
            "retains F'=0; the cooling-only specification had 23"
        ),
        "repumper_enabled": True,
        "gravity_included_in_force": False,
        "linewidth_rad_per_s": base.natural_linewidth_rad_per_s,
        "linewidth_hz": base.natural_linewidth_rad_per_s / (2.0 * np.pi),
        "limitations": [
            "The rate-equation model includes populations but excludes optical coherences and sub-Doppler physics.",
            "Turnaround is the global minimum within the configured positive-velocity window.",
            "Rows that fail coarse/fine convergence remain checkpointed and are visibly flagged in plots.",
        ],
        "outputs": {
            "force_sweep_csv": str(csv_path),
            "restoring_slope_plot": str(restoring_plot_path),
            "damping_turnaround_plot": str(turnaround_plot_path),
        },
    }


def run_force_detuning_sweep(
    *,
    detuning_n_values: Sequence[float] | None = None,
    numerics: ForceSweepNumerics | None = None,
    output_directory: Path | None = None,
    figure_directory: Path | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """Run, checkpoint, resume, and plot the multilevel detuning-force sweep."""

    values = _validate_detuning_values(
        DETUNING_N_VALUES if detuning_n_values is None else detuning_n_values
    )
    controls = numerics or ForceSweepNumerics()
    controls.validate()
    paths = multilevel_mot_paths()
    output = output_directory or paths["statistics"] / "force_vs_detuning"
    figures = figure_directory or paths["figures"] / "force_vs_detuning"
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
            raise ValueError(
                f"checkpoint point index does not match detuning n={row['detuning_n']}"
            )
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

    base = default_multilevel_mot_config()
    model = build_rate_equation_model(base.natural_linewidth_rad_per_s)
    coil = default_anti_helmholtz_config()
    started = perf_counter()
    completed_new_axes = 0
    total_new_axes = 3 * EVALUATION_COUNT * len(remaining)
    print(
        f"[force-sweep] points={len(values)}; resumed={resumed_point_count}; "
        f"remaining={len(remaining)}; deterministic evaluations/point={EVALUATION_COUNT}; "
        f"model_states={model.state_count}",
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
            f"[force-sweep] point {point_index+1}/{len(values)} start: "
            f"n={detuning_n:+.3f}, Delta/(2pi)="
            f"{detuning_n*base.natural_linewidth_rad_per_s/(2*np.pi*1e6):+.3f} MHz; "
            f"deterministic evaluations={EVALUATION_COUNT}",
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
                f"[force-sweep] point {point_index+1}/{len(values)} n={detuning_n:+.3f}; "
                f"axis={axis_label}; new-axes={completed_new_axes}/{total_new_axes}; "
                f"elapsed={elapsed:.1f} s; ETA={eta_s:.1f} s",
                flush=True,
            )

        row = evaluate_force_detuning_point(
            point_index,
            detuning_n,
            controls,
            model=model,
            coil_config=coil,
            axis_progress_callback=report_axis,
        )
        rows.append(row)
        rows.sort(key=lambda item: int(item["point_index"]))
        _write_csv_checkpoint(csv_path, rows)
        plot_restoring_slopes_vs_detuning(rows, restoring_plot_path)
        plot_damping_turnarounds_vs_detuning(rows, turnaround_plot_path)
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
            f"[force-sweep] point {point_index+1}/{len(values)} checkpointed: "
            f"n={detuning_n:+.3f}; all_converged={row['all_converged']}; "
            f"wall={row['point_wall_time_s']:.2f} s",
            flush=True,
        )

    if rows:
        plot_restoring_slopes_vs_detuning(rows, restoring_plot_path)
        plot_damping_turnarounds_vs_detuning(rows, turnaround_plot_path)
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
        f"[force-sweep] {status}: {len(rows)}/{len(values)} points; "
        f"all_converged={final_metadata['all_convergence_checks_passed']}; "
        f"wall={perf_counter()-started:.1f} s",
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
        description="Run the multilevel restoring/damping-force detuning sweep"
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
    numerics = ForceSweepNumerics(
        position_step_m=1.0e-3 * args.position_step_mm,
        velocity_extent_m_per_s=args.velocity_extent,
        velocity_step_m_per_s=args.velocity_step,
        restoring_relative_tolerance=args.restoring_relative_tolerance,
        turnaround_absolute_tolerance_m_per_s=args.turnaround_absolute_tolerance,
    )
    result = run_force_detuning_sweep(
        numerics=numerics,
        output_directory=args.output_dir,
        figure_directory=args.figures_dir,
        resume=args.resume,
    )
    print(json.dumps(result["outputs"], indent=2), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DETUNING_N_VALUES",
    "EVALUATION_COUNT",
    "REPLICATE_COUNT",
    "ForceSweepNumerics",
    "build_argument_parser",
    "build_force_sweep_configuration",
    "detuning_n_grid",
    "evaluate_force_detuning_point",
    "main",
    "plot_damping_turnarounds_vs_detuning",
    "plot_restoring_slopes_vs_detuning",
    "run_force_detuning_sweep",
]
