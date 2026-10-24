"""Recompute and plot current on-axis force/gradient model overlays.

All forces are evaluated live from the current implementations after the
mot_strange vector-conversion correction.  The atom is stationary, gravity is
excluded, and every displayed gradient is the numerical derivative of the
same force samples drawn immediately above it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.mot_strange.surrogate_effective_field import numerical_field_jacobian_t_per_m
from pmot.mot_strange.surrogate_effective_field import surrogate_effective_field_t
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pmot_vec_2Level import evaluate_pmot_vec_2level_local_observable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "updated_force_gradient_overlays_20261023"
)
AXES = ("x", "y", "z")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
# Stop at the outer edge of the 5-mm glass-wall bands.  The mot_strange
# surrogate is hard-clipped at +/-14.142 mm; extending beyond that numerical
# domain would introduce a discontinuous derivative spike that obscures the
# physically relevant comparison in the cell and wall region.
SCAN_LIMIT_MM = 12.5
COOLING_DETUNING_MHZ = -15.0
MULTILEVEL_COOLING_POWER_MW = 27.0
MULTILEVEL_REPUMP_POWER_MW = 0.1
PMOT_COOLING_POWER_MW = 1.0571
PMOT_TRAPPING_PATH_POWERS_MW = np.full(3, 31.524310589374092)
FIGURE_DPI = 220
GLASS_WALL_CENTERS_MM = (-10.0, 10.0)
GLASS_WALL_WIDTH_MM = 5.0

MODEL_LABELS = {
    "mot_multilevel": "Multilevel MOT",
    "mot_strange": "mot_strange",
    "pMOT_2Level": "2-level pMOT",
    "pmot_vec_2Level": "Vector-only 2-level pMOT",
}
MODEL_COLORS = {
    "mot_multilevel": "#2563eb",
    "mot_strange": "#d97706",
    "pMOT_2Level": "#be123c",
    "pmot_vec_2Level": "#0f766e",
}
MODEL_STYLES = {
    "mot_multilevel": "-",
    "mot_strange": "--",
    "pMOT_2Level": "-.",
    "pmot_vec_2Level": ":",
}


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise absolute."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def scan_coordinates_m(
    *,
    broad_samples: int = 301,
    center_samples: int = 161,
    focus_samples: int = 201,
) -> np.ndarray:
    """Use a broad grid plus dense origin/focus sampling for derivatives."""

    if min(broad_samples, center_samples, focus_samples) < 3:
        raise ValueError("all sampling counts must be at least three")
    broad_mm = np.linspace(-SCAN_LIMIT_MM, SCAN_LIMIT_MM, broad_samples)
    center_extent_mm = min(1.0, SCAN_LIMIT_MM)
    center_mm = np.linspace(-center_extent_mm, center_extent_mm, center_samples)
    focus_outer_mm = min(12.5, SCAN_LIMIT_MM)
    focus_inner_mm = min(7.5, focus_outer_mm)
    negative_focus_mm = np.linspace(
        -focus_outer_mm,
        -focus_inner_mm,
        focus_samples,
    )
    positive_focus_mm = np.linspace(
        focus_inner_mm,
        focus_outer_mm,
        focus_samples,
    )
    values_mm = np.unique(
        np.round(
            np.concatenate(
                (broad_mm, center_mm, negative_focus_mm, positive_focus_mm)
            ),
            decimals=10,
        )
    )
    return 1.0e-3 * values_mm


def multilevel_config():
    return replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=1.0e-3 * MULTILEVEL_COOLING_POWER_MW,
        repump_power_w_per_beam=1.0e-3 * MULTILEVEL_REPUMP_POWER_MW,
        cooling_detuning_rad_per_s=2.0 * np.pi * 1.0e6 * COOLING_DETUNING_MHZ,
        include_gravity=False,
    )


def pmot_environment():
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        include_gravity=False,
    )
    apparatus = default_simple_mot_apparatus()
    apparatus = replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            power_w_per_beam=1.0e-3 * PMOT_COOLING_POWER_MW,
            detuning_hz=1.0e6 * COOLING_DETUNING_MHZ,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    return (
        simple,
        cooling,
        build_pmot_2level_trapping_beams(),
        load_pmot_2level_polarizability_table(),
    )


def evaluate_models(scan_m: np.ndarray) -> pd.DataFrame:
    """Evaluate all four current force laws on identical on-axis positions."""

    ml_config = multilevel_config()
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=20.0)
    accelerated = prepare_accelerated(config=ml_config, coil=coil)
    rate_model = build_rate_equation_model()
    multilevel_beams = build_multilevel_mot_beams(config=ml_config)
    simple, cooling, trapping, polarizability = pmot_environment()
    trapping_powers_w = 1.0e-3 * PMOT_TRAPPING_PATH_POWERS_MW
    fallback_axis = (0.0, 0.0, 1.0)
    velocity = np.zeros(3)
    force_by_model = {
        model: {axis: np.empty(len(scan_m)) for axis in AXES}
        for model in MODEL_LABELS
    }
    start = perf_counter()

    for axis in AXES:
        component = AXIS_INDEX[axis]
        for index, coordinate_m in enumerate(scan_m):
            position = np.zeros(3)
            position[component] = coordinate_m

            conventional_force, _, _, _ = accelerated_observable(
                position,
                velocity,
                np.asarray(fallback_axis),
                accelerated,
            )
            force_by_model["mot_multilevel"][axis][index] = conventional_force[
                component
            ]

            strange = rate_equation_observable(
                rate_model,
                multilevel_beams,
                tuple(position),
                tuple(velocity),
                None,
                ml_config,
                fallback_axis,
                magnetic_field_function=surrogate_effective_field_t,
            )
            force_by_model["mot_strange"][axis][index] = strange.force_n[
                component
            ]

            full_pmot = evaluate_pmot_2level_local_observable(
                cooling,
                trapping,
                position,
                velocity,
                path_powers_w=trapping_powers_w,
                previous_axis=fallback_axis,
                simple_config=simple,
                polarizability_table=polarizability,
            )
            force_by_model["pMOT_2Level"][axis][index] = full_pmot.force_n[
                component
            ]

            vector_pmot = evaluate_pmot_vec_2level_local_observable(
                cooling,
                trapping,
                position,
                velocity,
                path_powers_w=trapping_powers_w,
                previous_axis=fallback_axis,
                simple_config=simple,
            )
            force_by_model["pmot_vec_2Level"][axis][index] = vector_pmot.force_n[
                component
            ]
        print(
            f"completed {axis}-axis ({len(scan_m)} points); "
            f"elapsed={perf_counter() - start:.1f} s",
            flush=True,
        )

    rows: list[dict[str, float | str]] = []
    for model, model_forces in force_by_model.items():
        for axis, force_n in model_forces.items():
            gradient_n_per_m = np.gradient(force_n, scan_m, edge_order=2)
            for coordinate_m, force, gradient in zip(
                scan_m,
                force_n,
                gradient_n_per_m,
                strict=True,
            ):
                rows.append(
                    {
                        "model": model,
                        "axis": axis,
                        "coordinate_mm": 1.0e3 * coordinate_m,
                        "force_n": force,
                        "force_zN": 1.0e21 * force,
                        "force_gradient_n_per_m": gradient,
                        "force_gradient_1e19_n_per_m": 1.0e19 * gradient,
                    }
                )
    return pd.DataFrame(rows)


def plot_overlay(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    *,
    title: str,
    output_path: Path,
) -> None:
    figure, panels = plt.subplots(2, 3, figsize=(17.2, 9.7), sharex="col")
    for column, axis in enumerate(AXES):
        for model in models:
            subset = frame[
                (frame["model"] == model) & (frame["axis"] == axis)
            ].sort_values("coordinate_mm")
            panels[0, column].plot(
                subset["coordinate_mm"],
                subset["force_zN"],
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                linewidth=2.0,
                label=MODEL_LABELS[model],
            )
            panels[1, column].plot(
                subset["coordinate_mm"],
                subset["force_gradient_1e19_n_per_m"],
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                linewidth=2.0,
                label=MODEL_LABELS[model],
            )

        other_axes = [other for other in AXES if other != axis]
        panels[0, column].set_title(
            f"Along {axis}-axis; {other_axes[0]}=0 mm, {other_axes[1]}=0 mm"
        )
        panels[0, column].set_ylabel(rf"$F_{axis}$ [zN]")
        panels[1, column].set_ylabel(
            rf"$\partial F_{axis}/\partial {axis}$ [$10^{{-19}}$ N/m]"
        )
        panels[1, column].set_xlabel(f"{axis} position [mm]")
        for row in range(2):
            panel = panels[row, column]
            panel.axhline(0.0, color="0.25", linewidth=0.8)
            panel.axvline(0.0, color="0.25", linewidth=0.8)
            for center in GLASS_WALL_CENTERS_MM:
                panel.axvspan(
                    center - 0.5 * GLASS_WALL_WIDTH_MM,
                    center + 0.5 * GLASS_WALL_WIDTH_MM,
                    color="#94a3b8",
                    alpha=0.16,
                    linewidth=0.0,
                )
            panel.set_xlim(-SCAN_LIMIT_MM, SCAN_LIMIT_MM)
            panel.grid(True, alpha=0.20)

    figure.suptitle(title, fontsize=17, y=0.988)
    figure.text(
        0.5,
        0.945,
        f"Stationary atom; gravity excluded; detuning = {COOLING_DETUNING_MHZ:g} MHz. "
        f"Multilevel MOT and mot_strange: cooling {MULTILEVEL_COOLING_POWER_MW:g} "
        f"mW/component, repump {MULTILEVEL_REPUMP_POWER_MW:g} mW/component. "
        f"Two-level pMOTs: cooling {PMOT_COOLING_POWER_MW:g} mW/component, "
        "trapping paths ("
        + ", ".join(f"{power:.3f}" for power in PMOT_TRAPPING_PATH_POWERS_MW)
        + ") mW; repump geometry passive.",
        ha="center",
        va="top",
        fontsize=10.8,
    )
    handles = [
        Line2D(
            (0,),
            (0,),
            color=MODEL_COLORS[model],
            linestyle=MODEL_STYLES[model],
            linewidth=2.2,
            label=MODEL_LABELS[model],
        )
        for model in models
    ]
    handles.append(
        Patch(
            facecolor="#94a3b8",
            alpha=0.20,
            label="glass-wall bands: centers +/-10 mm, width 5 mm",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=min(len(handles), 5),
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.085, 0.98, 0.91), h_pad=1.8, w_pad=1.4)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the current MOT, mot_strange, full two-level pMOT, and "
            "vector-only two-level pMOT on-axis force/gradient overlays."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=OUTPUT_DIRECTORY,
        help="Output directory (default: %(default)s).",
    )
    parser.add_argument(
        "--scan-limit-mm",
        type=float,
        default=SCAN_LIMIT_MM,
        help=(
            "Symmetric position limit in mm. Values beyond 14.142 mm cross the "
            "mot_strange hard clipping boundary (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--cooling-detuning-mhz",
        type=float,
        default=COOLING_DETUNING_MHZ,
        help="Cooling detuning used by all models in MHz (default: %(default)s).",
    )
    parser.add_argument(
        "--multilevel-cooling-power-mw",
        type=float,
        default=MULTILEVEL_COOLING_POWER_MW,
        help="MOT and mot_strange cooling power per component in mW.",
    )
    parser.add_argument(
        "--multilevel-repump-power-mw",
        type=float,
        default=MULTILEVEL_REPUMP_POWER_MW,
        help="MOT and mot_strange repump power per component in mW.",
    )
    parser.add_argument(
        "--pmot-cooling-power-mw",
        type=float,
        default=PMOT_COOLING_POWER_MW,
        help="Both two-level pMOT cooling powers per component in mW.",
    )
    parser.add_argument(
        "--pmot-trapping-powers-mw",
        type=float,
        nargs=3,
        metavar=("PX", "PY", "PZ"),
        default=tuple(PMOT_TRAPPING_PATH_POWERS_MW),
        help="Three pMOT trapping-path powers in mW (default: 31.524 each).",
    )
    parser.add_argument(
        "--broad-samples",
        type=int,
        default=301,
        help="Samples across the full scan before dense-grid merging.",
    )
    parser.add_argument(
        "--center-samples",
        type=int,
        default=161,
        help="Samples across the central +/-1 mm interval.",
    )
    parser.add_argument(
        "--focus-samples",
        type=int,
        default=201,
        help="Samples in each dense focus interval around +/-10 mm.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=FIGURE_DPI,
        help="PNG resolution in dots per inch (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    global SCAN_LIMIT_MM
    global COOLING_DETUNING_MHZ
    global MULTILEVEL_COOLING_POWER_MW
    global MULTILEVEL_REPUMP_POWER_MW
    global PMOT_COOLING_POWER_MW
    global PMOT_TRAPPING_PATH_POWERS_MW
    global FIGURE_DPI

    args = parse_args()
    if not np.isfinite(args.scan_limit_mm) or args.scan_limit_mm <= 0.0:
        raise ValueError("--scan-limit-mm must be finite and positive")
    powers = np.asarray(args.pmot_trapping_powers_mw, dtype=float)
    scalar_parameters = np.asarray(
        (
            args.multilevel_cooling_power_mw,
            args.multilevel_repump_power_mw,
            args.pmot_cooling_power_mw,
            args.cooling_detuning_mhz,
        ),
        dtype=float,
    )
    if not np.all(np.isfinite(scalar_parameters)):
        raise ValueError("powers and detuning must be finite")
    if np.any(scalar_parameters[:3] < 0.0) or np.any(~np.isfinite(powers)) or np.any(powers < 0.0):
        raise ValueError("all powers must be finite and non-negative")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    SCAN_LIMIT_MM = float(args.scan_limit_mm)
    COOLING_DETUNING_MHZ = float(args.cooling_detuning_mhz)
    MULTILEVEL_COOLING_POWER_MW = float(args.multilevel_cooling_power_mw)
    MULTILEVEL_REPUMP_POWER_MW = float(args.multilevel_repump_power_mw)
    PMOT_COOLING_POWER_MW = float(args.pmot_cooling_power_mw)
    PMOT_TRAPPING_PATH_POWERS_MW = powers
    FIGURE_DPI = int(args.dpi)

    output_directory = args.output_directory
    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory / "current_model_on_axis_force_gradient_linecuts.csv"
    metadata_path = output_directory / "metadata.json"
    scan_m = scan_coordinates_m(
        broad_samples=args.broad_samples,
        center_samples=args.center_samples,
        focus_samples=args.focus_samples,
    )
    frame = evaluate_models(scan_m)
    frame.to_csv(data_path, index=False)

    outputs = {
        "three_model": output_directory / "mot_strange_pmot2level_overlay.png",
        "four_model": output_directory / "mot_strange_pmot2level_vector_overlay.png",
        "pmot_pair": output_directory / "pmot2level_full_vs_vector_only_overlay.png",
    }
    plot_overlay(
        frame,
        ("mot_multilevel", "mot_strange", "pMOT_2Level"),
        title="Force and Force-Gradient Overlay: MOT, mot_strange, and 2-Level pMOT",
        output_path=outputs["three_model"],
    )
    plot_overlay(
        frame,
        (
            "mot_multilevel",
            "mot_strange",
            "pMOT_2Level",
            "pmot_vec_2Level",
        ),
        title="Force and Force-Gradient Overlay: All Four Current Models",
        output_path=outputs["four_model"],
    )
    plot_overlay(
        frame,
        ("pMOT_2Level", "pmot_vec_2Level"),
        title="Force and Force-Gradient Overlay: Full vs Vector-Only 2-Level pMOT",
        output_path=outputs["pmot_pair"],
    )

    strange_gradient = 100.0 * np.diag(
        numerical_field_jacobian_t_per_m(step_m=1.0e-7)
    )
    origin = frame[np.isclose(frame["coordinate_mm"], 0.0)].copy()
    metadata = {
        "scan_limit_mm": SCAN_LIMIT_MM,
        "sample_count_per_axis": int(len(scan_m)),
        "stationary_atom": True,
        "gravity_excluded": True,
        "cooling_detuning_mhz": COOLING_DETUNING_MHZ,
        "multilevel_mot": {
            "cooling_power_mw_per_component": MULTILEVEL_COOLING_POWER_MW,
            "repump_power_mw_per_component": MULTILEVEL_REPUMP_POWER_MW,
            "field_gradient_g_per_cm_xyz": [10.0, 10.0, -20.0],
        },
        "mot_strange": {
            "cooling_power_mw_per_component": MULTILEVEL_COOLING_POWER_MW,
            "repump_power_mw_per_component": MULTILEVEL_REPUMP_POWER_MW,
            "corrected_surrogate_field_gradient_g_per_cm_xyz": strange_gradient.tolist(),
            "propagated_trapping_beams": 0,
        },
        "two_level_pmots": {
            "cooling_power_mw_per_component": PMOT_COOLING_POWER_MW,
            "passive_repump_power_mw_per_component": MULTILEVEL_REPUMP_POWER_MW,
            "trapping_path_powers_mw_xyz": PMOT_TRAPPING_PATH_POWERS_MW.tolist(),
        },
        "origin_force_gradient_1e19_n_per_m": {
            row["model"] + "_" + row["axis"]: row[
                "force_gradient_1e19_n_per_m"
            ]
            for _, row in origin.iterrows()
        },
        "data_csv": display_path(data_path),
        "figures": {
            name: display_path(path) for name, path in outputs.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
