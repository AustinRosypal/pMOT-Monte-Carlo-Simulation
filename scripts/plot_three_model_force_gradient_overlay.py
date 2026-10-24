"""Create matched 2-by-3 force/gradient panels for all three MOT models.

The top row shows the stationary same-axis radiation-pressure force along x,
y, and z.  The bottom row shows the direct numerical derivative of the force
displayed immediately above it.  Gravity is excluded.  The regular MOT and
``mot_strange`` use the replacement 24-state population-rate solver; the
``pMOT_2Level`` curve is explicitly provisional.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_strange.surrogate_effective_field import (
    surrogate_field_jacobians_t_per_m,
)
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_multilevel_and_strange_force_gradient_corrected_ranges_20261020"
    / "extended_force_gradient_linecuts.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "three_model_force_gradient_overlay_20261020"
)

SCAN_LIMIT_MM = 15.0
OFFSETS_MM = (0.0, 1.0, 2.0)
TRAPPING_PATH_POWERS_W = (19.147243e-3, 19.147243e-3, 38.294486e-3)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
CUT_SPECS = (
    ("x", "y", "z"),
    ("y", "x", "z"),
    ("z", "x", "y"),
)
MODEL_ORDER = ("mot_multilevel", "mot_strange", "pMOT_2Level")
MODEL_LABELS = {
    "mot_multilevel": "Multilevel MOT",
    "mot_strange": "mot_strange",
    "pMOT_2Level": "2-level pMOT",
}
MODEL_COLORS = {
    "mot_multilevel": "#2563eb",
    "mot_strange": "#d97706",
    "pMOT_2Level": "#be123c",
}
MODEL_LINESTYLES = {
    "mot_multilevel": "-",
    "mot_strange": "--",
    "pMOT_2Level": "-.",
}
GLASS_CENTERS_MM = (-10.0, 10.0)
GLASS_WIDTH_MM = 5.0
GLASS_COLOR = "#64748b"


def _position(
    coordinate_mm: float,
    scan_axis: str,
    transverse_axis: str,
    fixed_axis: str,
    offset_mm: float,
) -> np.ndarray:
    result = np.zeros(3, dtype=float)
    result[AXIS_INDEX[scan_axis]] = 1.0e-3 * coordinate_mm
    result[AXIS_INDEX[transverse_axis]] = 1.0e-3 * offset_mm
    result[AXIS_INDEX[fixed_axis]] = 0.0
    return result


def _source_frame() -> pd.DataFrame:
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(
            f"Missing corrected multilevel/strange-MOT source data: {SOURCE_CSV}"
        )
    frame = pd.read_csv(SOURCE_CSV)
    return frame[np.abs(frame["scan_coordinate_mm"]) <= SCAN_LIMIT_MM].copy()


def _coordinate_grid(
    source: pd.DataFrame,
    *,
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
) -> np.ndarray:
    selector = (
        (source["model"] == "mot_strange")
        & (source["scan_axis"] == scan_axis)
        & (source["transverse_axis"] == transverse_axis)
        & (source["transverse_offset_mm"] == offset_mm)
    )
    coordinates = np.sort(
        source.loc[selector, "scan_coordinate_mm"].to_numpy(dtype=float)
    )
    if len(coordinates) < 3:
        raise RuntimeError(
            "The corrected strange-MOT source does not contain a complete "
            f"{scan_axis}/{transverse_axis}/{offset_mm:g} mm line cut."
        )
    return coordinates


def _evaluate_pmot_2level(source: pd.DataFrame) -> pd.DataFrame:
    simple = replace(default_simple_mot_config(), include_gravity=False)
    cooling = build_pmot_2level_cooling_beams(simple_config=simple)
    trapping = build_pmot_2level_trapping_beams()
    polarizability = load_pmot_2level_polarizability_table()
    rows: list[dict[str, object]] = []
    total = sum(
        len(
            _coordinate_grid(
                source,
                scan_axis=scan_axis,
                transverse_axis=transverse_axis,
                offset_mm=offset_mm,
            )
        )
        for scan_axis, transverse_axis, _ in CUT_SPECS
        for offset_mm in OFFSETS_MM
    )
    completed = 0
    start = perf_counter()

    for scan_axis, transverse_axis, fixed_axis in CUT_SPECS:
        component = AXIS_INDEX[scan_axis]
        for offset_mm in OFFSETS_MM:
            coordinates_mm = _coordinate_grid(
                source,
                scan_axis=scan_axis,
                transverse_axis=transverse_axis,
                offset_mm=offset_mm,
            )
            forces_n = np.empty(len(coordinates_mm), dtype=float)
            previous_axis = np.array((0.0, 0.0, 1.0))
            for index, coordinate_mm in enumerate(coordinates_mm):
                position_m = _position(
                    coordinate_mm,
                    scan_axis,
                    transverse_axis,
                    fixed_axis,
                    offset_mm,
                )
                sample = evaluate_pmot_2level_local_observable(
                    cooling,
                    trapping,
                    position_m,
                    np.zeros(3),
                    path_powers_w=TRAPPING_PATH_POWERS_W,
                    previous_axis=previous_axis,
                    simple_config=simple,
                    polarizability_table=polarizability,
                )
                forces_n[index] = sample.force_n[component]
                if sample.stark.quantization_axis_from_vector_shift:
                    previous_axis = np.asarray(
                        sample.stark.quantization_axis,
                        dtype=float,
                    )
                completed += 1
                if completed == 1 or completed % 500 == 0 or completed == total:
                    print(
                        f"[2-level pMOT] samples {completed}/{total}; "
                        f"elapsed={perf_counter() - start:.1f} s",
                        flush=True,
                    )

            gradients_n_per_m = np.gradient(
                forces_n,
                1.0e-3 * coordinates_mm,
                edge_order=2,
            )
            for coordinate_mm, force_n, gradient_n_per_m in zip(
                coordinates_mm,
                forces_n,
                gradients_n_per_m,
                strict=True,
            ):
                rows.append(
                    {
                        "model": "pMOT_2Level",
                        "scan_axis": scan_axis,
                        "transverse_axis": transverse_axis,
                        "fixed_axis": fixed_axis,
                        "transverse_offset_mm": offset_mm,
                        "scan_coordinate_mm": coordinate_mm,
                        "same_axis_force_n": force_n,
                        "same_axis_force_zN": 1.0e21 * force_n,
                        "same_axis_force_gradient_n_per_m": gradient_n_per_m,
                        "same_axis_force_gradient_1e19_n_per_m": (
                            1.0e19 * gradient_n_per_m
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_comparison_frame(
    output_directory: Path,
    *,
    reuse_data: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Return the matched three-model line cuts, computing pMOT data if needed."""

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "three_model_force_gradient_linecuts.csv"
    if reuse_data and csv_path.is_file():
        return pd.read_csv(csv_path), csv_path

    source = _source_frame()
    pmot = _evaluate_pmot_2level(source)
    frame = pd.concat((source, pmot), ignore_index=True, sort=False)
    frame = frame[frame["model"].isin(MODEL_ORDER)].copy()
    frame.to_csv(csv_path, index=False)
    return frame, csv_path


def _cut_frame(
    frame: pd.DataFrame,
    *,
    model: str,
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
) -> pd.DataFrame:
    subset = frame[
        (frame["model"] == model)
        & (frame["scan_axis"] == scan_axis)
        & (frame["transverse_axis"] == transverse_axis)
        & (frame["transverse_offset_mm"] == offset_mm)
        & (np.abs(frame["scan_coordinate_mm"]) <= SCAN_LIMIT_MM)
    ].sort_values("scan_coordinate_mm")
    if subset.empty:
        raise RuntimeError(
            f"Missing line cut for {model}/{scan_axis}/{transverse_axis}/"
            f"{offset_mm:g} mm."
        )
    return subset


def _offset_label(offset_mm: float) -> tuple[str, str]:
    if offset_mm == 0.0:
        return "On-axis", "on_axis"
    return f"+{offset_mm:g} mm transverse offset", f"offset_{offset_mm:g}mm"


def _decorate(axis, *, shade_glass: bool) -> None:
    if shade_glass:
        for center_mm in GLASS_CENTERS_MM:
            axis.axvspan(
                center_mm - 0.5 * GLASS_WIDTH_MM,
                center_mm + 0.5 * GLASS_WIDTH_MM,
                color=GLASS_COLOR,
                alpha=0.10,
                zorder=0,
            )
            axis.axvline(
                center_mm,
                color="0.35",
                linestyle=":",
                linewidth=0.9,
                zorder=1,
            )
    axis.axhline(0.0, color="0.25", linewidth=0.85, alpha=0.8)
    axis.axvline(0.0, color="0.25", linewidth=0.85, alpha=0.8)
    axis.grid(True, which="major", alpha=0.22)
    axis.grid(True, which="minor", alpha=0.08)
    axis.minorticks_on()
    axis.set_xlim(-SCAN_LIMIT_MM, SCAN_LIMIT_MM)


def plot_force_gradient_panels(
    frame: pd.DataFrame,
    *,
    models: tuple[str, ...],
    offset_mm: float,
    output_path: Path,
    title: str,
    shade_glass: bool,
) -> Path:
    """Plot force above its derivative for the x, y, and z line cuts."""

    figure, axes = plt.subplots(2, 3, figsize=(17.0, 9.2), sharex="col")
    for column, (scan_axis, transverse_axis, fixed_axis) in enumerate(CUT_SPECS):
        for model in models:
            cut = _cut_frame(
                frame,
                model=model,
                scan_axis=scan_axis,
                transverse_axis=transverse_axis,
                offset_mm=offset_mm,
            )
            plot_kwargs = {
                "color": MODEL_COLORS[model],
                "linestyle": MODEL_LINESTYLES[model],
                "linewidth": 2.1,
                "label": MODEL_LABELS[model],
            }
            axes[0, column].plot(
                cut["scan_coordinate_mm"],
                cut["same_axis_force_zN"],
                **plot_kwargs,
            )
            axes[1, column].plot(
                cut["scan_coordinate_mm"],
                cut["same_axis_force_gradient_1e19_n_per_m"],
                **plot_kwargs,
            )

        axes[0, column].set_title(
            f"Along {scan_axis}-axis\n"
            f"{transverse_axis}={offset_mm:+g} mm, {fixed_axis}=0 mm"
        )
        axes[0, column].set_ylabel(rf"$F_{scan_axis}$ [zN]")
        axes[1, column].set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ "
            r"[$10^{-19}$ N/m]"
        )
        axes[1, column].set_xlabel(f"{scan_axis} [mm]")
        _decorate(axes[0, column], shade_glass=shade_glass)
        _decorate(axes[1, column], shade_glass=shade_glass)

    offset_text, _ = _offset_label(offset_mm)
    figure.suptitle(title, fontsize=17, y=0.995)
    figure.text(
        0.5,
        0.963,
        f"{offset_text}; stationary atom; radiation pressure only; gravity excluded",
        ha="center",
        va="top",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.936,
        "MOT coil and 2-level pMOT target: "
        r"$(\partial_xB_x,\partial_yB_y,\partial_zB_z)=(+10,+10,-20)$ G/cm; "
        "mot_strange field is spatially nonuniform",
        ha="center",
        va="top",
        fontsize=11.5,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS[model],
            linestyle=MODEL_LINESTYLES[model],
            linewidth=2.3,
            label=MODEL_LABELS[model],
        )
        for model in models
    ]
    if shade_glass:
        legend_handles.append(
            Patch(
                facecolor=GLASS_COLOR,
                alpha=0.18,
                label="pMOT optical glass-cell wall regions (center ±10 mm, width 5 mm)",
            )
        )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=min(len(legend_handles), 4),
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.89), h_pad=1.5, w_pad=1.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _origin_field_gradients_g_per_cm() -> dict[str, list[float]]:
    strange_jacobian = surrogate_field_jacobians_t_per_m(
        np.zeros(3),
        step_m=1.0e-6,
    )
    return {
        "mot_multilevel": [10.0, 10.0, -20.0],
        "mot_strange": (100.0 * np.diag(strange_jacobian)).tolist(),
        "pMOT_2Level": [10.0, 10.0, -20.0],
    }


def run(
    output_directory: Path = DEFAULT_OUTPUT,
    *,
    reuse_data: bool = True,
    include_individual: bool = True,
) -> dict[str, object]:
    """Generate overlay figures and optionally the matched individual figures."""

    output_directory = output_directory.resolve()
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    frame, csv_path = build_comparison_frame(
        output_directory,
        reuse_data=reuse_data,
    )

    overlay_paths: list[Path] = []
    individual_paths: dict[str, list[Path]] = {
        model: [] for model in MODEL_ORDER
    }
    for offset_mm in OFFSETS_MM:
        _, suffix = _offset_label(offset_mm)
        overlay_paths.append(
            plot_force_gradient_panels(
                frame,
                models=MODEL_ORDER,
                offset_mm=offset_mm,
                output_path=(
                    figure_directory
                    / f"three_model_force_gradient_overlay_{suffix}.png"
                ),
                title="Three-Model MOT Force and Same-Axis Force-Gradient Comparison",
                shade_glass=True,
            )
        )
        if include_individual:
            for model in MODEL_ORDER:
                path = plot_force_gradient_panels(
                    frame,
                    models=(model,),
                    offset_mm=offset_mm,
                    output_path=(
                        figure_directory
                        / f"{model}_force_gradient_{suffix}.png"
                    ),
                    title=(
                        f"{MODEL_LABELS[model]}: Force and Same-Axis Force Gradient"
                    ),
                    shade_glass=model != "mot_multilevel",
                )
                individual_paths[model].append(path)

    simple = replace(default_simple_mot_config(), include_gravity=False)
    manifest = {
        "schema": "pmot.three-model-force-gradient-overlay.v1",
        "scientific_status": (
            "diagnostic comparison; the 2-level pMOT is provisional and is not "
            "a validated physical pMOT prediction"
        ),
        "sampling": {
            "scan_range_mm": [-SCAN_LIMIT_MM, SCAN_LIMIT_MM],
            "transverse_offsets_mm": list(OFFSETS_MM),
            "stationary_atom": True,
            "gravity_included": False,
            "force_gradient_definition": (
                "numpy.gradient of each displayed same-axis force versus its "
                "saved scan coordinate, edge_order=2"
            ),
        },
        "models": {
            "mot_multilevel": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "field_gradient_g_per_cm": [10.0, 10.0, -20.0],
            },
            "mot_strange": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "field": "centered intensity-derived surrogate magnetic field",
                "origin_field_gradient_g_per_cm": (
                    _origin_field_gradients_g_per_cm()["mot_strange"]
                ),
            },
            "pMOT_2Level": {
                "solver": "provisional effective F=2 to F'=3 saturated closure",
                "simple_configuration": asdict(simple),
                "cooling_power_mw_per_beam": 27.0,
                "cooling_detuning_mhz": -15.0,
                "trapping_path_powers_mw": [
                    1.0e3 * value for value in TRAPPING_PATH_POWERS_W
                ],
                "vector_shift_gradient_target_g_per_cm": [10.0, 10.0, -20.0],
                "rate_normalization": (
                    "ordinary-frequency linewidth with scattering prefactor "
                    "Gamma/2=pi*gamma_nu (2*pi correction included)"
                ),
            },
        },
        "source_csv": str(SOURCE_CSV.relative_to(PROJECT_ROOT)),
        "outputs": {
            "linecuts_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "overlays": [
                str(path.relative_to(PROJECT_ROOT)) for path in overlay_paths
            ],
            "individual": {
                model: [str(path.relative_to(PROJECT_ROOT)) for path in paths]
                for model, paths in individual_paths.items()
            },
        },
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_directory": output_directory,
        "linecuts_csv": csv_path,
        "manifest": manifest_path,
        "overlay_figures": overlay_paths,
        "individual_figures": individual_paths,
    }


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="recompute the 2-level pMOT line cuts instead of reusing saved data",
    )
    parser.add_argument(
        "--overlays-only",
        action="store_true",
        help="omit the matching individual-model figures",
    )
    args = parser.parse_args()
    result = run(
        args.output,
        reuse_data=not args.recompute,
        include_individual=not args.overlays_only,
    )
    print(json.dumps(_json_ready(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
