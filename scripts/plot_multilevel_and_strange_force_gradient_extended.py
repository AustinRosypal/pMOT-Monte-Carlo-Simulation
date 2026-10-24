"""Create extended force/gradient figures for the MOT and ``mot_strange``.

The figures use stationary atoms, exclude gravity, and evaluate three
principal-axis cuts at transverse offsets of 0, +1, and +2 mm.  Each figure
places the same-axis force above its direct numerical same-axis derivative.
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

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_multilevel.accelerated import observable as accelerated_observable
from pmot.mot_multilevel.accelerated import prepare as prepare_accelerated
from pmot.mot_strange.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)
from pmot.mot_strange.surrogate_effective_field import (
    surrogate_effective_field_t,
)
from pmot.mot_strange.surrogate_effective_field import (
    surrogate_field_jacobians_t_per_m,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRANGE_SOURCE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_strange"
    / "three_model_force_field_gradient_comparison_20261016"
    / "gradient_linecuts.csv"
)
MULTILEVEL_SOURCE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_multilevel"
    / "force_gradient_cartesian_planes_15_20gpcm_20261019"
    / "multilevel_mot_force_gradient_linecuts_15_20gpcm.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_multilevel_and_strange_force_gradient_corrected_ranges_20261020"
)

SCAN_LIMIT_MM = {
    "mot_multilevel": 25.0,
    "mot_strange": 15.0,
}
INNER_LIMIT_MM = 12.0
OFFSETS_MM = (0.0, 1.0, 2.0)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
CUT_SPECS = (
    ("x", "y", "z"),
    ("y", "x", "z"),
    ("z", "x", "y"),
)
MODELS = ("mot_multilevel", "mot_strange")
SOURCE_MODEL = {
    "mot_multilevel": "multilevel_coil",
    "mot_strange": "multilevel_surrogate",
}
MODEL_TITLE = {
    "mot_multilevel": "Regular Multilevel MOT",
    "mot_strange": "mot_strange: Multilevel MOT with Intensity-Derived Field",
}
FORCE_COLOR = "#1261a0"
GRADIENT_COLOR = "#d95f02"
GLASS_COLOR = "#64748b"
GLASS_CENTERS_MM = (-10.0, 10.0)
GLASS_WIDTH_MM = 5.0


def _outer_coordinates_mm(limit_mm: float) -> np.ndarray:
    negative = np.arange(-limit_mm, -INNER_LIMIT_MM, 0.2)
    positive = np.arange(INNER_LIMIT_MM + 0.2, limit_mm + 0.1, 0.2)
    return np.unique(np.round(np.concatenate((negative, positive)), decimals=9))


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


def _position_key(position_m: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(position_m, decimals=12))


def _outer_positions(coordinates_mm: np.ndarray) -> np.ndarray:
    keyed: dict[tuple[float, float, float], np.ndarray] = {}
    for scan_axis, transverse_axis, fixed_axis in CUT_SPECS:
        for offset_mm in OFFSETS_MM:
            for coordinate_mm in coordinates_mm:
                position = _position(
                    coordinate_mm,
                    scan_axis,
                    transverse_axis,
                    fixed_axis,
                    offset_mm,
                )
                keyed[_position_key(position)] = position
    return np.asarray([keyed[key] for key in sorted(keyed)])


def _evaluate_outer_forces(
    multilevel_positions_m: np.ndarray,
    strange_positions_m: np.ndarray,
) -> dict[str, dict[tuple[float, float, float], np.ndarray]]:
    config = replace(default_multilevel_mot_config(), include_gravity=False)
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=20.0)
    accelerated = prepare_accelerated(config=config, coil=coil)
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    velocity = np.zeros(3)
    fallback_axis = np.array((0.0, 0.0, 1.0))
    result = {model_name: {} for model_name in MODELS}
    start = perf_counter()

    for index, position in enumerate(multilevel_positions_m, start=1):
        key = _position_key(position)
        conventional_force, _, _, _ = accelerated_observable(
            position,
            velocity,
            fallback_axis,
            accelerated,
        )
        result["mot_multilevel"][key] = np.asarray(
            conventional_force,
            dtype=float,
        )
        if index == 1 or index % 200 == 0 or index == len(multilevel_positions_m):
            print(
                f"[extended line cuts] multilevel outer positions "
                f"{index}/{len(multilevel_positions_m)}; "
                f"elapsed={perf_counter() - start:.1f} s",
                flush=True,
            )

    strange_start = perf_counter()
    for index, position in enumerate(strange_positions_m, start=1):
        key = _position_key(position)
        strange = rate_equation_observable(
            model,
            beams,
            key,
            (0.0, 0.0, 0.0),
            None,
            config,
            tuple(fallback_axis),
            magnetic_field_function=surrogate_effective_field_t,
        )
        result["mot_strange"][key] = np.asarray(strange.force_n, dtype=float)
        if index == 1 or index % 200 == 0 or index == len(strange_positions_m):
            print(
                f"[extended line cuts] strange outer positions "
                f"{index}/{len(strange_positions_m)}; "
                f"elapsed={perf_counter() - strange_start:.1f} s",
                flush=True,
            )
    return result


def _inner_force(
    sources: dict[str, pd.DataFrame],
    *,
    model_name: str,
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
) -> pd.DataFrame:
    plane = "".join(sorted((scan_axis, transverse_axis)))
    source = sources[model_name]
    selector = (
        (source["plane"] == plane)
        & (source["scan_axis"] == scan_axis)
        & (source["transverse_axis"] == transverse_axis)
        & (source["transverse_offset_mm"] == offset_mm)
    )
    if model_name == "mot_multilevel":
        selector &= source["axial_gradient_g_per_cm"] == 20.0
    else:
        selector &= source["model"] == SOURCE_MODEL[model_name]
    subset = source[selector][
        ["scan_coordinate_mm", "same_axis_force_n"]
    ].copy()
    if subset.empty:
        raise RuntimeError(
            f"Missing source line cut for {model_name}/{scan_axis}/"
            f"{transverse_axis}/{offset_mm:g} mm"
        )
    return subset


def _assemble_extended_frame(
    sources: dict[str, pd.DataFrame],
    outer_coordinates_mm: dict[str, np.ndarray],
    outer_forces: dict[str, dict[tuple[float, float, float], np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in MODELS:
        for scan_axis, transverse_axis, fixed_axis in CUT_SPECS:
            component = AXIS_INDEX[scan_axis]
            for offset_mm in OFFSETS_MM:
                inner = _inner_force(
                    sources,
                    model_name=model_name,
                    scan_axis=scan_axis,
                    transverse_axis=transverse_axis,
                    offset_mm=offset_mm,
                )
                coordinate_mm = inner["scan_coordinate_mm"].to_numpy(dtype=float)
                force_n = inner["same_axis_force_n"].to_numpy(dtype=float)
                outer_values = []
                model_outer_coordinates = outer_coordinates_mm[model_name]
                for coordinate in model_outer_coordinates:
                    position = _position(
                        coordinate,
                        scan_axis,
                        transverse_axis,
                        fixed_axis,
                        offset_mm,
                    )
                    outer_values.append(
                        outer_forces[model_name][_position_key(position)][component]
                    )
                coordinate_mm = np.concatenate(
                    (coordinate_mm, model_outer_coordinates)
                )
                force_n = np.concatenate((force_n, np.asarray(outer_values)))
                order = np.argsort(coordinate_mm)
                coordinate_mm = coordinate_mm[order]
                force_n = force_n[order]
                gradient_n_per_m = np.gradient(
                    force_n,
                    1.0e-3 * coordinate_mm,
                    edge_order=2,
                )
                for coordinate, force, gradient in zip(
                    coordinate_mm,
                    force_n,
                    gradient_n_per_m,
                    strict=True,
                ):
                    rows.append(
                        {
                            "model": model_name,
                            "scan_axis": scan_axis,
                            "transverse_axis": transverse_axis,
                            "fixed_axis": fixed_axis,
                            "transverse_offset_mm": offset_mm,
                            "scan_coordinate_mm": coordinate,
                            "same_axis_force_n": force,
                            "same_axis_force_zN": 1.0e21 * force,
                            "same_axis_force_gradient_n_per_m": gradient,
                            "same_axis_force_gradient_1e19_n_per_m": (
                                1.0e19 * gradient
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _origin_field_gradients() -> dict[str, np.ndarray]:
    coil_axial = 20.0
    conventional = np.asarray((0.5 * coil_axial, 0.5 * coil_axial, -coil_axial))
    strange_jacobian = surrogate_field_jacobians_t_per_m(
        np.zeros(3),
        step_m=1.0e-6,
    )
    strange = 100.0 * np.diag(strange_jacobian)
    return {
        "mot_multilevel": conventional,
        "mot_strange": strange,
    }


def _cut_frame(
    frame: pd.DataFrame,
    *,
    model_name: str,
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
) -> pd.DataFrame:
    return frame[
        (frame["model"] == model_name)
        & (frame["scan_axis"] == scan_axis)
        & (frame["transverse_axis"] == transverse_axis)
        & (frame["transverse_offset_mm"] == offset_mm)
    ].sort_values("scan_coordinate_mm")


def _decorate_axis(axis, *, shade_glass: bool) -> None:
    if shade_glass:
        for center_mm in GLASS_CENTERS_MM:
            axis.axvspan(
                center_mm - 0.5 * GLASS_WIDTH_MM,
                center_mm + 0.5 * GLASS_WIDTH_MM,
                color=GLASS_COLOR,
                alpha=0.12,
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


def _offset_label(offset_mm: float) -> tuple[str, str]:
    if offset_mm == 0.0:
        return "On-axis", "on_axis"
    return f"+{offset_mm:g} mm transverse offset", f"offset_{offset_mm:g}mm"


def _plot_figure(
    frame: pd.DataFrame,
    *,
    model_name: str,
    offset_mm: float,
    field_gradients_g_per_cm: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17.0, 9.2), sharex="col")
    shade_glass = model_name == "mot_strange"

    for column, (scan_axis, transverse_axis, fixed_axis) in enumerate(CUT_SPECS):
        cut = _cut_frame(
            frame,
            model_name=model_name,
            scan_axis=scan_axis,
            transverse_axis=transverse_axis,
            offset_mm=offset_mm,
        )
        coordinate = cut["scan_coordinate_mm"]
        axes[0, column].plot(
            coordinate,
            cut["same_axis_force_zN"],
            color=FORCE_COLOR,
            linewidth=2.2,
        )
        axes[1, column].plot(
            coordinate,
            cut["same_axis_force_gradient_1e19_n_per_m"],
            color=GRADIENT_COLOR,
            linewidth=2.0,
        )
        axes[0, column].set_ylabel(rf"$F_{scan_axis}$ [zN]")
        axes[1, column].set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ "
            r"[$10^{-19}$ N/m]"
        )
        axes[1, column].set_xlabel(f"{scan_axis} [mm]")
        axes[0, column].set_title(
            f"Along {scan_axis}-axis\n"
            f"{transverse_axis}={offset_mm:+g} mm, {fixed_axis}=0 mm"
        )
        _decorate_axis(axes[0, column], shade_glass=shade_glass)
        _decorate_axis(axes[1, column], shade_glass=shade_glass)
        axes[0, column].set_xlim(
            -SCAN_LIMIT_MM[model_name],
            SCAN_LIMIT_MM[model_name],
        )
        axes[1, column].set_xlim(
            -SCAN_LIMIT_MM[model_name],
            SCAN_LIMIT_MM[model_name],
        )

    offset_text, _ = _offset_label(offset_mm)
    gx, gy, gz = field_gradients_g_per_cm
    if model_name == "mot_multilevel":
        field_text = (
            "Anti-Helmholtz field gradient at origin: "
            rf"$(\partial_xB_x,\partial_yB_y,\partial_zB_z)="
            rf"({gx:+.3f},{gy:+.3f},{gz:+.3f})$ G/cm; "
            r"axial magnitude $|\partial_zB_z|=20$ G/cm"
        )
    else:
        field_text = (
            "Effective magnetic-field gradient at origin: "
            r"$(\partial_x B_{\mathrm{eff},x},\partial_y B_{\mathrm{eff},y},"
            r"\partial_z B_{\mathrm{eff},z})="
            f"({gx:+.3f},{gy:+.3f},{gz:+.3f})$ G/cm; spatially nonuniform"
        )

    figure.suptitle(
        f"{MODEL_TITLE[model_name]}: Force and Same-Axis Force Gradient",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.963,
        f"{offset_text}; stationary atom; gravity excluded",
        ha="center",
        va="top",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.937,
        field_text,
        ha="center",
        va="top",
        fontsize=11.5,
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=FORCE_COLOR,
            linewidth=2.2,
            label="same-axis force (top row)",
        ),
        Line2D(
            [0],
            [0],
            color=GRADIENT_COLOR,
            linewidth=2.0,
            label="numerical derivative of displayed force (bottom row)",
        ),
    ]
    if shade_glass:
        handles.append(
            Patch(
                facecolor=GLASS_COLOR,
                alpha=0.20,
                label="glass-cell walls (center ±10 mm, width 5 mm)",
            )
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(handles),
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.89), h_pad=1.5, w_pad=1.5)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run(output_directory: Path, *, reuse_data: bool = False) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    figure_directory = output_directory / "figures"
    figure_directory.mkdir(exist_ok=True)
    csv_path = output_directory / "extended_force_gradient_linecuts.csv"

    if reuse_data and csv_path.is_file():
        frame = pd.read_csv(csv_path)
        print(f"Reusing {csv_path}", flush=True)
    else:
        sources = {
            "mot_multilevel": pd.read_csv(MULTILEVEL_SOURCE_CSV),
            "mot_strange": pd.read_csv(STRANGE_SOURCE_CSV),
        }
        outer_coordinates_mm = {
            model_name: _outer_coordinates_mm(SCAN_LIMIT_MM[model_name])
            for model_name in MODELS
        }
        positions_m = {
            model_name: _outer_positions(outer_coordinates_mm[model_name])
            for model_name in MODELS
        }
        print(
            f"Evaluating {len(positions_m['mot_multilevel'])} multilevel and "
            f"{len(positions_m['mot_strange'])} strange-MOT unique outer "
            f"positions beyond ±{INNER_LIMIT_MM:g} mm",
            flush=True,
        )
        outer_forces = _evaluate_outer_forces(
            positions_m["mot_multilevel"],
            positions_m["mot_strange"],
        )
        frame = _assemble_extended_frame(
            sources,
            outer_coordinates_mm,
            outer_forces,
        )
        frame.to_csv(csv_path, index=False)

    field_gradients = _origin_field_gradients()
    figures: dict[str, list[str]] = {model_name: [] for model_name in MODELS}
    for model_name in MODELS:
        for offset_mm in OFFSETS_MM:
            _, suffix = _offset_label(offset_mm)
            limit_label = f"plus_minus_{SCAN_LIMIT_MM[model_name]:g}mm"
            path = figure_directory / (
                f"{model_name}_force_gradient_{limit_label}_{suffix}.png"
            )
            _plot_figure(
                frame,
                model_name=model_name,
                offset_mm=offset_mm,
                field_gradients_g_per_cm=field_gradients[model_name],
                output_path=path,
            )
            figures[model_name].append(str(path.relative_to(PROJECT_ROOT)))

    config = replace(default_multilevel_mot_config(), include_gravity=False)
    manifest = {
        "schema": "pmot.multilevel-and-strange-extended-force-gradients.v1",
        "scientific_status": {
            "mot_multilevel": (
                "stationary local-force diagnostic from the replacement physical "
                "24-state Section-12 solver"
            ),
            "mot_strange": (
                "diagnostic use of the same 24-state solver with the historical "
                "intensity-derived surrogate effective magnetic field"
            ),
        },
        "configuration": asdict(config),
        "sampling": {
            "range_mm": {
                model_name: [-limit, limit]
                for model_name, limit in SCAN_LIMIT_MM.items()
            },
            "offsets_mm": list(OFFSETS_MM),
            "stationary_atom": True,
            "gravity_included": False,
            "cuts": [
                {
                    "scan_axis": scan,
                    "transverse_axis": transverse,
                    "fixed_axis": fixed,
                }
                for scan, transverse, fixed in CUT_SPECS
            ],
            "force_gradient": (
                "numpy.gradient of each displayed same-axis force curve over the "
                "saved nonuniform coordinate grid, edge_order=2"
            ),
        },
        "origin_field_gradient_g_per_cm": {
            key: value.tolist() for key, value in field_gradients.items()
        },
        "strange_field_configuration": (
            DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.metadata()
        ),
        "glass_wall_annotation": {
            "models": ["mot_strange"],
            "centers_mm": list(GLASS_CENTERS_MM),
            "width_mm": GLASS_WIDTH_MM,
        },
        "outputs": {
            "linecuts_csv": str(csv_path.relative_to(PROJECT_ROOT)),
            "figures": figures,
        },
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_directory": str(output_directory),
        "linecuts_csv": str(csv_path),
        "manifest": str(manifest_path),
        "figures": figures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), reuse_data=args.reuse_data), indent=2))


if __name__ == "__main__":
    main()
