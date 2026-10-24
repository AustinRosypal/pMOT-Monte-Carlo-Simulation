"""Compare local force and effective-field gradients for three MOT models.

The comparison contains:

1. the replacement 24-state Section-12 MOT with the shared 10 G/cm
   anti-Helmholtz coil;
2. the same 24-state solver with the centered intensity-derived surrogate
   magnetic field from :mod:`pmot.mot_strange.surrogate_effective_field`;
3. the provisional effective two-level pMOT with six 1529-nm trapping
   components and no external field.

For each Cartesian scan axis, line cuts are evaluated in both cardinal planes
containing that axis.  The transverse offsets are 0, 1, 2, and 12 mm.  Gravity
is excluded, the atom is stationary, and the plotted quantities are the
same-axis derivatives dF_i/dx_i and dB_i/dx_i.

This is a diagnostic comparison.  In particular, the two-level pMOT does not
share the 24-state dissipative kernel used by the first two models, and its
effective field is only the stretched-transition-equivalent representation of
the vector Stark-energy vector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmot.magnetic_fields import anti_helmholtz_field_t
from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_multilevel.atomic_structure import BOHR_MAGNETON_J_PER_T
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_differential_ac_stark_shift
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.mot_strange.surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
)
from pmot.mot_strange.surrogate_effective_field import surrogate_effective_field_t


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_strange"
    / "three_model_force_field_gradient_comparison_20261016"
)

AXES = "xyz"
AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}
TRANSVERSE_AXES = {
    "x": ("y", "z"),
    "y": ("x", "z"),
    "z": ("x", "y"),
}
OFFSETS_MM = (0.0, 1.0, 2.0, 12.0)
SCAN_LIMIT_MM = 12.0
TRAPPING_POWER_W_PER_PATH = 38.294486e-3
PMOT_TRANSITION_ZEEMAN_FACTOR = 1.0

MODEL_LABELS = {
    "multilevel_coil": "24-state MOT, 10 G/cm coil",
    "multilevel_surrogate": "24-state MOT, intensity-derived B",
    "pmot_2level": "effective two-level pMOT",
}
MODEL_COLORS = {
    "multilevel_coil": "#2563eb",
    "multilevel_surrogate": "#d97706",
    "pmot_2level": "#be123c",
}
MODEL_STYLES = {
    "multilevel_coil": "--",
    "multilevel_surrogate": "-.",
    "pmot_2level": "-",
}


def _scan_coordinates_m() -> np.ndarray:
    """Resolve the central volume and the narrow +/-10-mm waist regions."""

    broad = np.linspace(-SCAN_LIMIT_MM, SCAN_LIMIT_MM, 121)
    central = np.linspace(-1.0, 1.0, 81)
    negative_focus = np.linspace(-10.15, -9.85, 41)
    positive_focus = np.linspace(9.85, 10.15, 41)
    values_mm = np.unique(
        np.round(
            np.concatenate((broad, central, negative_focus, positive_focus, [0.0])),
            decimals=9,
        )
    )
    return 1.0e-3 * values_mm


def _line_positions(
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
    scan_m: np.ndarray,
) -> np.ndarray:
    positions = np.zeros((len(scan_m), 3), dtype=float)
    positions[:, AXIS_INDEX[scan_axis]] = scan_m
    positions[:, AXIS_INDEX[transverse_axis]] = 1.0e-3 * offset_mm
    return positions


def _coil_fields_t(positions_m: np.ndarray, coil) -> np.ndarray:
    values = anti_helmholtz_field_t(
        positions_m[:, 0],
        positions_m[:, 1],
        positions_m[:, 2],
        coil,
    )
    return np.column_stack(values)


def _gradient(values: np.ndarray, coordinates_m: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=float), coordinates_m, edge_order=2)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _build_contexts():
    multilevel_config = replace(default_multilevel_mot_config(), include_gravity=False)
    multilevel_model = build_rate_equation_model()
    multilevel_beams = build_multilevel_mot_beams(config=multilevel_config)
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=10.0)

    simple_config = replace(default_simple_mot_config(), include_gravity=False)
    pmot_cooling = build_pmot_2level_cooling_beams(simple_config=simple_config)
    pmot_trapping = build_pmot_2level_trapping_beams()
    polarizability_table = load_pmot_2level_polarizability_table()
    return {
        "multilevel_config": multilevel_config,
        "multilevel_model": multilevel_model,
        "multilevel_beams": multilevel_beams,
        "coil": coil,
        "simple_config": simple_config,
        "pmot_cooling": pmot_cooling,
        "pmot_trapping": pmot_trapping,
        "polarizability_table": polarizability_table,
    }


def _position_key(position_m: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.round(position_m, decimals=12))


def _evaluate_unique_positions(
    unique_positions_m: np.ndarray,
    contexts,
) -> dict[str, dict[tuple[float, float, float], np.ndarray]]:
    """Evaluate all force and field vectors once per unique position."""

    coil = contexts["coil"]
    result = {
        model: {"force": {}, "field": {}}
        for model in MODEL_LABELS
    }
    coil_fields = _coil_fields_t(unique_positions_m, coil)
    surrogate_fields = np.asarray(surrogate_effective_field_t(unique_positions_m), dtype=float)

    for position, coil_field, surrogate_field in zip(
        unique_positions_m, coil_fields, surrogate_fields
    ):
        key = _position_key(position)
        result["multilevel_coil"]["field"][key] = coil_field
        result["multilevel_surrogate"]["field"][key] = surrogate_field

    total = len(unique_positions_m)
    start = perf_counter()
    for index, position in enumerate(unique_positions_m, start=1):
        key = _position_key(position)
        coil_observable = rate_equation_observable(
            contexts["multilevel_model"],
            contexts["multilevel_beams"],
            key,
            (0.0, 0.0, 0.0),
            contexts["coil"],
            contexts["multilevel_config"],
        )
        surrogate_observable = rate_equation_observable(
            contexts["multilevel_model"],
            contexts["multilevel_beams"],
            key,
            (0.0, 0.0, 0.0),
            None,
            contexts["multilevel_config"],
            magnetic_field_function=surrogate_effective_field_t,
        )
        pmot_observable = evaluate_pmot_2level_local_observable(
            contexts["pmot_cooling"],
            contexts["pmot_trapping"],
            position,
            np.zeros(3),
            path_powers_w=(TRAPPING_POWER_W_PER_PATH,) * 3,
            simple_config=contexts["simple_config"],
            polarizability_table=contexts["polarizability_table"],
        )
        result["multilevel_coil"]["force"][key] = np.asarray(
            coil_observable.force_n, dtype=float
        )
        result["multilevel_surrogate"]["force"][key] = np.asarray(
            surrogate_observable.force_n, dtype=float
        )
        result["pmot_2level"]["force"][key] = np.asarray(
            pmot_observable.force_n, dtype=float
        )
        vector_energy_j = np.asarray(
            pmot_observable.stark.net_vector_energy_vector_j,
            dtype=float,
        )
        result["pmot_2level"]["field"][key] = vector_energy_j / (
            BOHR_MAGNETON_J_PER_T * PMOT_TRANSITION_ZEEMAN_FACTOR
        )
        if index == 1 or index % 250 == 0 or index == total:
            elapsed = perf_counter() - start
            print(
                f"[gradient comparison] positions {index}/{total}; "
                f"elapsed={elapsed:.1f} s",
                flush=True,
            )
    return result


def _all_unique_positions(scan_m: np.ndarray) -> np.ndarray:
    positions = []
    for scan_axis in AXES:
        for transverse_axis in TRANSVERSE_AXES[scan_axis]:
            for offset_mm in OFFSETS_MM:
                positions.extend(
                    _line_positions(scan_axis, transverse_axis, offset_mm, scan_m)
                )
    keyed = {_position_key(np.asarray(position)): np.asarray(position) for position in positions}
    return np.asarray([keyed[key] for key in sorted(keyed)])


def _assemble_frame(scan_m: np.ndarray, samples) -> pd.DataFrame:
    rows = []
    for scan_axis in AXES:
        component = AXIS_INDEX[scan_axis]
        for transverse_axis in TRANSVERSE_AXES[scan_axis]:
            plane = "".join(sorted((scan_axis, transverse_axis)))
            for offset_mm in OFFSETS_MM:
                positions = _line_positions(
                    scan_axis,
                    transverse_axis,
                    offset_mm,
                    scan_m,
                )
                keys = [_position_key(position) for position in positions]
                for model_name in MODEL_LABELS:
                    force_n = np.asarray(
                        [samples[model_name]["force"][key][component] for key in keys]
                    )
                    field_t = np.asarray(
                        [samples[model_name]["field"][key][component] for key in keys]
                    )
                    force_gradient_n_per_m = _gradient(force_n, scan_m)
                    field_gradient_t_per_m = _gradient(field_t, scan_m)
                    for index, coordinate_m in enumerate(scan_m):
                        rows.append(
                            {
                                "model": model_name,
                                "model_label": MODEL_LABELS[model_name],
                                "plane": plane,
                                "scan_axis": scan_axis,
                                "transverse_axis": transverse_axis,
                                "transverse_offset_mm": offset_mm,
                                "scan_coordinate_mm": 1.0e3 * coordinate_m,
                                "same_axis_force_n": force_n[index],
                                "same_axis_force_zN": 1.0e21 * force_n[index],
                                "same_axis_force_gradient_n_per_m": force_gradient_n_per_m[index],
                                "same_axis_force_gradient_1e19_n_per_m": (
                                    1.0e19 * force_gradient_n_per_m[index]
                                ),
                                "same_axis_effective_field_t": field_t[index],
                                "same_axis_effective_field_g": 1.0e4 * field_t[index],
                                "same_axis_field_gradient_t_per_m": field_gradient_t_per_m[index],
                                "same_axis_field_gradient_g_per_cm": (
                                    100.0 * field_gradient_t_per_m[index]
                                ),
                            }
                        )
    return pd.DataFrame(rows)


def _plot_plane(
    frame: pd.DataFrame,
    *,
    scan_axis: str,
    transverse_axis: str,
    output_path: Path,
    central_zoom: bool,
) -> None:
    figure, axes = plt.subplots(
        len(OFFSETS_MM),
        2,
        figsize=(13.2, 12.8),
        sharex=True,
        constrained_layout=True,
    )
    plane = "".join(sorted((scan_axis, transverse_axis)))
    for row, offset_mm in enumerate(OFFSETS_MM):
        subset = frame[
            (frame["scan_axis"] == scan_axis)
            & (frame["transverse_axis"] == transverse_axis)
            & (frame["transverse_offset_mm"] == offset_mm)
        ]
        if central_zoom:
            subset = subset[np.abs(subset["scan_coordinate_mm"]) <= 3.0]
        for model_name in MODEL_LABELS:
            model = subset[subset["model"] == model_name]
            axes[row, 0].plot(
                model["scan_coordinate_mm"],
                model["same_axis_field_gradient_g_per_cm"],
                color=MODEL_COLORS[model_name],
                linestyle=MODEL_STYLES[model_name],
                linewidth=1.7,
                label=MODEL_LABELS[model_name],
            )
            axes[row, 1].plot(
                model["scan_coordinate_mm"],
                model["same_axis_force_gradient_1e19_n_per_m"],
                color=MODEL_COLORS[model_name],
                linestyle=MODEL_STYLES[model_name],
                linewidth=1.7,
                label=MODEL_LABELS[model_name],
            )
        axes[row, 0].set_ylabel(
            rf"$\partial B_{scan_axis}/\partial {scan_axis}$ [G/cm]"
            + "\n"
            + rf"${transverse_axis}={offset_mm:g}$ mm"
        )
        axes[row, 1].set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ [$10^{{-19}}$ N/m]"
            + "\n"
            + rf"${transverse_axis}={offset_mm:g}$ mm"
        )
        for panel in axes[row]:
            panel.axhline(0.0, color="0.35", linewidth=0.7)
            panel.axvline(0.0, color="0.35", linewidth=0.7)
            panel.grid(alpha=0.2)
            if not central_zoom:
                panel.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    axes[0, 0].set_title("Same-axis effective magnetic-field gradient")
    axes[0, 1].set_title("Stationary radiation-force gradient")
    axes[-1, 0].set_xlabel(f"{scan_axis} [mm]")
    axes[-1, 1].set_xlabel(f"{scan_axis} [mm]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        ncol=3,
        frameon=False,
    )
    scale_label = "central +/-3 mm, linear scale" if central_zoom else "full +/-12 mm, symlog scale"
    figure.suptitle(
        f"{plane.upper()} plane: gradients along {scan_axis} at {transverse_axis} offsets "
        f"({scale_label})\nstationary atom; gravity excluded"
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_principal_axes(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    central_zoom: bool,
) -> None:
    """Overlay the three models on the zero-offset principal-axis cuts."""

    figure, axes = plt.subplots(
        3,
        2,
        figsize=(13.2, 9.5),
        sharex=True,
        constrained_layout=True,
    )
    for row, scan_axis in enumerate(AXES):
        transverse_axis = TRANSVERSE_AXES[scan_axis][0]
        subset = frame[
            (frame["scan_axis"] == scan_axis)
            & (frame["transverse_axis"] == transverse_axis)
            & (frame["transverse_offset_mm"] == 0.0)
        ]
        if central_zoom:
            subset = subset[np.abs(subset["scan_coordinate_mm"]) <= 3.0]
        for model_name in MODEL_LABELS:
            model = subset[subset["model"] == model_name]
            axes[row, 0].plot(
                model["scan_coordinate_mm"],
                model["same_axis_field_gradient_g_per_cm"],
                color=MODEL_COLORS[model_name],
                linestyle=MODEL_STYLES[model_name],
                linewidth=1.8,
                label=MODEL_LABELS[model_name],
            )
            axes[row, 1].plot(
                model["scan_coordinate_mm"],
                model["same_axis_force_gradient_1e19_n_per_m"],
                color=MODEL_COLORS[model_name],
                linestyle=MODEL_STYLES[model_name],
                linewidth=1.8,
                label=MODEL_LABELS[model_name],
            )
        axes[row, 0].set_ylabel(
            rf"$\partial B_{scan_axis}/\partial {scan_axis}$ [G/cm]"
        )
        axes[row, 1].set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ [$10^{{-19}}$ N/m]"
        )
        for panel in axes[row]:
            panel.axhline(0.0, color="0.35", linewidth=0.7)
            panel.axvline(0.0, color="0.35", linewidth=0.7)
            panel.grid(alpha=0.2)
            if not central_zoom:
                panel.set_yscale("symlog", linthresh=0.1, linscale=0.8)
    axes[0, 0].set_title("Same-axis effective magnetic-field gradient")
    axes[0, 1].set_title("Stationary radiation-force gradient")
    axes[-1, 0].set_xlabel("scan coordinate [mm]")
    axes[-1, 1].set_xlabel("scan coordinate [mm]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
    )
    scale_label = "central +/-3 mm, linear scale" if central_zoom else "full +/-12 mm, symlog scale"
    figure.suptitle(
        f"Zero-offset principal-axis gradient comparison ({scale_label})\n"
        "stationary atom; gravity excluded"
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _central_summary(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model_name in MODEL_LABELS:
        summary[model_name] = {}
        for scan_axis in AXES:
            transverse_axis = TRANSVERSE_AXES[scan_axis][0]
            subset = frame[
                (frame["model"] == model_name)
                & (frame["scan_axis"] == scan_axis)
                & (frame["transverse_axis"] == transverse_axis)
                & (frame["transverse_offset_mm"] == 0.0)
            ].sort_values("scan_coordinate_mm")
            origin_index = np.argmin(np.abs(subset["scan_coordinate_mm"].to_numpy()))
            origin = subset.iloc[int(origin_index)]
            central = subset[np.abs(subset["scan_coordinate_mm"]) <= 2.0]
            full = subset[np.abs(subset["scan_coordinate_mm"]) <= SCAN_LIMIT_MM]
            summary[model_name][scan_axis] = {
                "origin_field_gradient_g_per_cm": float(
                    origin["same_axis_field_gradient_g_per_cm"]
                ),
                "origin_force_gradient_1e19_n_per_m": float(
                    origin["same_axis_force_gradient_1e19_n_per_m"]
                ),
                "maximum_absolute_field_gradient_within_2mm_g_per_cm": float(
                    central["same_axis_field_gradient_g_per_cm"].abs().max()
                ),
                "maximum_absolute_force_gradient_within_2mm_1e19_n_per_m": float(
                    central["same_axis_force_gradient_1e19_n_per_m"].abs().max()
                ),
                "maximum_absolute_field_gradient_full_scan_g_per_cm": float(
                    full["same_axis_field_gradient_g_per_cm"].abs().max()
                ),
                "maximum_absolute_force_gradient_full_scan_1e19_n_per_m": float(
                    full["same_axis_force_gradient_1e19_n_per_m"].abs().max()
                ),
                "restoring_gradient_fraction_within_2mm": float(
                    np.mean(central["same_axis_force_gradient_n_per_m"] < 0.0)
                ),
            }
    return summary


def run(output_directory: Path, *, reuse_data: bool = False) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = output_directory / "figures"
    figures.mkdir(exist_ok=True)
    scan_m = _scan_coordinates_m()
    contexts = _build_contexts()
    csv_path = output_directory / "gradient_linecuts.csv"
    if reuse_data and csv_path.is_file():
        frame = pd.read_csv(csv_path)
        scan_m = 1.0e-3 * np.sort(frame["scan_coordinate_mm"].unique())
        print(f"[gradient comparison] reusing {csv_path}", flush=True)
    else:
        unique_positions = _all_unique_positions(scan_m)
        print(
            f"[gradient comparison] scan samples={len(scan_m)}, "
            f"unique 3D positions={len(unique_positions)}",
            flush=True,
        )
        samples = _evaluate_unique_positions(unique_positions, contexts)
        frame = _assemble_frame(scan_m, samples)
        frame.to_csv(csv_path, index=False)

    plot_paths = []
    for scan_axis in AXES:
        for transverse_axis in TRANSVERSE_AXES[scan_axis]:
            plane = "".join(sorted((scan_axis, transverse_axis)))
            for central_zoom in (False, True):
                suffix = "central_zoom" if central_zoom else "full_scan"
                path = figures / (
                    f"{plane}_scan_{scan_axis}_offset_{transverse_axis}_{suffix}.png"
                )
                _plot_plane(
                    frame,
                    scan_axis=scan_axis,
                    transverse_axis=transverse_axis,
                    output_path=path,
                    central_zoom=central_zoom,
                )
                plot_paths.append(path)

    for central_zoom in (False, True):
        suffix = "central_zoom" if central_zoom else "full_scan"
        path = figures / f"principal_axes_zero_offset_{suffix}.png"
        _plot_principal_axes(
            frame,
            output_path=path,
            central_zoom=central_zoom,
        )
        plot_paths.append(path)

    summary = _central_summary(frame)
    summary_path = output_directory / "gradient_summary.json"
    summary_path.write_text(
        json.dumps(_json_ready(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "pmot.three-model-force-field-gradient-comparison.v1",
        "scientific_status": "diagnostic comparison; not a validated pMOT prediction",
        "models": {
            "multilevel_coil": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "field": "shared exact anti-Helmholtz field",
                "coil_target_axial_gradient_g_per_cm": 10.0,
                "configuration": asdict(contexts["multilevel_config"]),
            },
            "multilevel_surrogate": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "field": "centered intensity-derived stretched-transition surrogate",
                "configuration": DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.metadata(),
            },
            "pmot_2level": {
                "solver": "provisional deterministic effective F=2 to F'=3 closure",
                "field_definition": (
                    "net vector Stark-energy vector divided by mu_B times G=1; "
                    "scalar and tensor shifts affect force but are excluded from B_eff"
                ),
                "trapping_power_w_per_path": TRAPPING_POWER_W_PER_PATH,
                "trapping_wavelength_nm": 1529.268881,
                "cooling_power_w_per_beam": 27.0e-3,
                "cooling_detuning_hz": -15.0e6,
                "helicities_xyz_propagation_frame": ["sigma+", "sigma+", "sigma-"],
            },
        },
        "sampling": {
            "scan_limit_mm": SCAN_LIMIT_MM,
            "scan_sample_count": len(scan_m),
            "transverse_offsets_mm": OFFSETS_MM,
            "planes_per_scan_axis": TRANSVERSE_AXES,
            "stationary_atom": True,
            "gravity_included": False,
            "field_gradient": "numerical dB_i/dx_i on the nonuniform saved scan grid",
            "force_gradient": "numerical dF_i/dx_i on the nonuniform saved scan grid",
        },
        "outputs": {
            "linecut_csv": csv_path,
            "summary_json": summary_path,
            "figures": plot_paths,
        },
        "comparison_caveats": [
            "The first two force curves use the same physical 24-state kernel; the pMOT force curve uses the effective two-level saturated closure.",
            "The intensity-derived field and two-level pMOT use different historical optical geometries and powers.",
            "The pMOT B_eff curve represents only the vector Stark energy; its scalar and tensor shifts remain in its force calculation.",
            "Negative dF_i/dx_i is locally restoring; red detuning alone is not sufficient for restoration.",
        ],
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "output_directory": output_directory,
        "csv": csv_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "figures": plot_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help="regenerate plots and summaries from an existing gradient_linecuts.csv",
    )
    args = parser.parse_args()
    result = run(args.output.resolve(), reuse_data=args.reuse_data)
    print(json.dumps(_json_ready(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
