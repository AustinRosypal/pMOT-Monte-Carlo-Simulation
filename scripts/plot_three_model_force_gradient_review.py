"""Create matched force/gradient diagnostics for the three current MOT models.

The line cuts use stationary atoms, exclude gravity, and scan the six ordered
Cartesian cuts (x and y in XY, x and z in XZ, y and z in YZ) at positive
in-plane transverse offsets of 0, 1, and 2 mm.  Every displayed force gradient
is the numerical derivative of the displayed same-axis force samples.

The conventional multilevel reference uses the saved 20 G/cm axial-coil data.
The ``mot_strange`` samples come from the saved replacement-24-state evaluation
with the centered intensity-derived field.  The ``pMOT_2Level`` samples are
recomputed at the selected low-saturation setting: 1.0571 mW per cooling
traveling component, -15 MHz detuning, and trapping-path powers
(19.147243, 19.147243, 38.294486) mW, corresponding to the provisional
(+10,+10,-20) G/cm vector-shift target.
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
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MULTILEVEL_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_multilevel"
    / "force_gradient_cartesian_planes_15_20gpcm_20261019"
    / "multilevel_mot_force_gradient_linecuts_15_20gpcm.csv"
)
MOT_STRANGE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "mot_strange"
    / "three_model_force_field_gradient_comparison_20261016"
    / "gradient_linecuts.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "diagnostics"
    / "three_model_force_gradient_review_pmot_cooling_1p0571mW_20261023"
)

OFFSETS_MM = (0.0, 1.0, 2.0)
SCAN_LIMIT_MM = 12.0
TRAPPING_PATH_POWERS_W = (19.147243e-3, 19.147243e-3, 38.294486e-3)
PMOT_COOLING_POWER_MW_PER_BEAM = 1.0571
PANEL_SPECS = (
    ("xy", "x", "y"),
    ("xy", "y", "x"),
    ("xz", "x", "z"),
    ("xz", "z", "x"),
    ("yz", "y", "z"),
    ("yz", "z", "y"),
)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

MODEL_LABELS = {
    "mot_multilevel": "Multilevel MOT, (+10,+10,-20) G/cm",
    "mot_strange": "mot_strange, intensity-derived field",
    "pMOT_2Level": "2-level pMOT, 1.0571 mW/beam (s0 = 1.000)",
}
MODEL_COLORS = {
    "mot_multilevel": "#2563eb",
    "mot_strange": "#d97706",
    "pMOT_2Level": "#be123c",
}
MODEL_STYLES = {
    "mot_multilevel": "--",
    "mot_strange": "-.",
    "pMOT_2Level": "-",
}
FORCE_COLOR = "#1261a0"
GRADIENT_COLOR = "#d95f02"
GLASS_WALL_CENTERS_MM = (-10.0, 10.0)
GLASS_WALL_WIDTH_MM = 5.0
GLASS_WALL_COLOR = "#64748b"


def _scan_coordinates_m() -> np.ndarray:
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


def _force_gradient(force_n: np.ndarray, scan_m: np.ndarray) -> np.ndarray:
    return np.gradient(np.asarray(force_n, dtype=float), scan_m, edge_order=2)


def _load_multilevel_reference() -> pd.DataFrame:
    if not MULTILEVEL_CSV.is_file():
        raise FileNotFoundError(f"Missing multilevel reference data: {MULTILEVEL_CSV}")
    frame = pd.read_csv(MULTILEVEL_CSV)
    frame = frame[
        (frame["axial_gradient_g_per_cm"] == 20.0)
        & frame["transverse_offset_mm"].isin(OFFSETS_MM)
    ].copy()
    frame["model"] = "mot_multilevel"
    frame["model_label"] = MODEL_LABELS["mot_multilevel"]
    return frame


def _load_mot_strange_reference() -> pd.DataFrame:
    if not MOT_STRANGE_CSV.is_file():
        raise FileNotFoundError(f"Missing mot_strange reference data: {MOT_STRANGE_CSV}")
    frame = pd.read_csv(MOT_STRANGE_CSV)
    frame = frame[
        (frame["model"] == "multilevel_surrogate")
        & frame["transverse_offset_mm"].isin(OFFSETS_MM)
    ].copy()
    frame["model"] = "mot_strange"
    frame["model_label"] = MODEL_LABELS["mot_strange"]
    return frame


def _evaluate_pmot_2level(scan_m: np.ndarray) -> pd.DataFrame:
    simple = replace(default_simple_mot_config(), include_gravity=False)
    baseline_apparatus = default_simple_mot_apparatus()
    apparatus = replace(
        baseline_apparatus,
        cooling=replace(
            baseline_apparatus.cooling,
            power_w_per_beam=1.0e-3 * PMOT_COOLING_POWER_MW_PER_BEAM,
            detuning_hz=simple.cooling_detuning_hz,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    trapping = build_pmot_2level_trapping_beams()
    polarizability = load_pmot_2level_polarizability_table()
    rows: list[dict[str, object]] = []
    total = len(PANEL_SPECS) * len(OFFSETS_MM) * len(scan_m)
    completed = 0
    start = perf_counter()

    for plane, scan_axis, transverse_axis in PANEL_SPECS:
        component = AXIS_INDEX[scan_axis]
        for offset_mm in OFFSETS_MM:
            positions = _line_positions(scan_axis, transverse_axis, offset_mm, scan_m)
            force_n = np.empty(len(scan_m), dtype=float)
            previous_axis = np.array([0.0, 0.0, 1.0])
            for index, position in enumerate(positions):
                observable = evaluate_pmot_2level_local_observable(
                    cooling,
                    trapping,
                    position,
                    np.zeros(3),
                    path_powers_w=TRAPPING_PATH_POWERS_W,
                    previous_axis=previous_axis,
                    simple_config=simple,
                    polarizability_table=polarizability,
                )
                force_n[index] = observable.force_n[component]
                if observable.stark.quantization_axis_from_vector_shift:
                    previous_axis = np.asarray(
                        observable.stark.quantization_axis,
                        dtype=float,
                    )
                completed += 1
                if completed == 1 or completed % 500 == 0 or completed == total:
                    print(
                        f"[pMOT_2Level] samples {completed}/{total}; "
                        f"elapsed={perf_counter() - start:.1f} s",
                        flush=True,
                    )
            gradient = _force_gradient(force_n, scan_m)
            for index, coordinate_m in enumerate(scan_m):
                rows.append(
                    {
                        "model": "pMOT_2Level",
                        "model_label": MODEL_LABELS["pMOT_2Level"],
                        "plane": plane,
                        "scan_axis": scan_axis,
                        "transverse_axis": transverse_axis,
                        "transverse_offset_mm": offset_mm,
                        "scan_coordinate_mm": 1.0e3 * coordinate_m,
                        "same_axis_force_n": force_n[index],
                        "same_axis_force_zN": 1.0e21 * force_n[index],
                        "same_axis_force_gradient_n_per_m": gradient[index],
                        "same_axis_force_gradient_1e19_n_per_m": 1.0e19
                        * gradient[index],
                    }
                )
    return pd.DataFrame(rows)


def _panel_data(
    frame: pd.DataFrame,
    *,
    model: str,
    plane: str,
    scan_axis: str,
    transverse_axis: str,
    offset_mm: float,
) -> pd.DataFrame:
    subset = frame[
        (frame["model"] == model)
        & (frame["plane"] == plane)
        & (frame["scan_axis"] == scan_axis)
        & (frame["transverse_axis"] == transverse_axis)
        & (frame["transverse_offset_mm"] == offset_mm)
    ].sort_values("scan_coordinate_mm")
    if len(subset) != 271:
        raise RuntimeError(
            f"Expected 271 samples for {model}/{plane}/{scan_axis}/"
            f"{transverse_axis}/{offset_mm:g} mm; found {len(subset)}"
        )
    return subset


def _cut_description(scan_axis: str, transverse_axis: str, offset_mm: float) -> str:
    fixed = {"x": 0.0, "y": 0.0, "z": 0.0}
    fixed[transverse_axis] = offset_mm
    return ", ".join(
        f"{axis}={fixed[axis]:+g} mm" for axis in "xyz" if axis != scan_axis
    )


def _offset_text(offset_mm: float) -> tuple[str, str]:
    if offset_mm == 0.0:
        return "On-axis", "on_axis"
    return (
        f"+{offset_mm:g} mm in-plane transverse offset",
        f"offset_{offset_mm:g}mm",
    )


def _plot_force_and_gradient(
    frame: pd.DataFrame,
    *,
    model: str,
    offset_mm: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 13.5), sharex=True)
    for panel_axis, (plane, scan_axis, transverse_axis) in zip(
        axes.flat, PANEL_SPECS, strict=True
    ):
        subset = _panel_data(
            frame,
            model=model,
            plane=plane,
            scan_axis=scan_axis,
            transverse_axis=transverse_axis,
            offset_mm=offset_mm,
        )
        coordinate = subset["scan_coordinate_mm"].to_numpy(dtype=float)
        force = subset["same_axis_force_zN"].to_numpy(dtype=float)
        gradient = subset["same_axis_force_gradient_1e19_n_per_m"].to_numpy(
            dtype=float
        )

        panel_axis.axhline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
        panel_axis.axvline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
        panel_axis.plot(coordinate, force, color=FORCE_COLOR, linewidth=2.2)
        gradient_axis = panel_axis.twinx()
        gradient_axis.plot(
            coordinate,
            gradient,
            color=GRADIENT_COLOR,
            linewidth=2.0,
            linestyle="--",
        )
        gradient_axis.axhline(0.0, color="0.25", linewidth=0.7, alpha=0.45)
        panel_axis.set_title(
            f"{plane.upper()} plane: along {scan_axis}-axis\n"
            f"{_cut_description(scan_axis, transverse_axis, offset_mm)}",
            fontsize=11.5,
        )
        panel_axis.set_ylabel(rf"$F_{scan_axis}$ [zN]", color=FORCE_COLOR)
        gradient_axis.set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ "
            r"[$10^{-19}$ N/m]",
            color=GRADIENT_COLOR,
        )
        panel_axis.tick_params(axis="y", colors=FORCE_COLOR)
        gradient_axis.tick_params(axis="y", colors=GRADIENT_COLOR)
        panel_axis.grid(True, which="major", alpha=0.22)
        panel_axis.grid(True, which="minor", alpha=0.08)
        panel_axis.minorticks_on()
        panel_axis.set_xlim(-SCAN_LIMIT_MM, SCAN_LIMIT_MM)
        panel_axis.set_xlabel(f"{scan_axis} [mm]")

    offset_label, _ = _offset_text(offset_mm)
    figure.suptitle(
        f"{MODEL_LABELS[model]}: Force and Same-Axis Force Gradient",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.969,
        f"{offset_label}; stationary atom; gravity excluded",
        ha="center",
        va="top",
        fontsize=13,
    )
    figure.legend(
        handles=(
            Line2D([0], [0], color=FORCE_COLOR, linewidth=2.2, label="same-axis force"),
            Line2D(
                [0],
                [0],
                color=GRADIENT_COLOR,
                linewidth=2.0,
                linestyle="--",
                label="numerical derivative of the displayed force",
            ),
        ),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.006),
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.925), h_pad=2.2, w_pad=2.4)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_gradient_overlay(
    frame: pd.DataFrame,
    *,
    offset_mm: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(15.5, 12.8), sharex=True)
    for panel_axis, (plane, scan_axis, transverse_axis) in zip(
        axes.flat, PANEL_SPECS, strict=True
    ):
        for center_mm in GLASS_WALL_CENTERS_MM:
            panel_axis.axvspan(
                center_mm - 0.5 * GLASS_WALL_WIDTH_MM,
                center_mm + 0.5 * GLASS_WALL_WIDTH_MM,
                color=GLASS_WALL_COLOR,
                alpha=0.10,
                zorder=0,
            )
            panel_axis.axvline(
                center_mm,
                color="0.35",
                linestyle=":",
                linewidth=0.9,
                zorder=1,
            )
        for model in MODEL_LABELS:
            subset = _panel_data(
                frame,
                model=model,
                plane=plane,
                scan_axis=scan_axis,
                transverse_axis=transverse_axis,
                offset_mm=offset_mm,
            )
            panel_axis.plot(
                subset["scan_coordinate_mm"],
                subset["same_axis_force_gradient_1e19_n_per_m"],
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                linewidth=2.0,
                label=MODEL_LABELS[model],
            )
        panel_axis.axhline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
        panel_axis.axvline(0.0, color="0.25", linewidth=0.9, alpha=0.75)
        panel_axis.set_title(
            f"{plane.upper()} plane: along {scan_axis}-axis\n"
            f"{_cut_description(scan_axis, transverse_axis, offset_mm)}",
            fontsize=11.5,
        )
        panel_axis.set_ylabel(
            rf"$\partial F_{scan_axis}/\partial {scan_axis}$ "
            r"[$10^{-19}$ N/m]"
        )
        panel_axis.set_xlabel(f"{scan_axis} [mm]")
        panel_axis.set_xlim(
            -10.0 - 0.5 * GLASS_WALL_WIDTH_MM,
            10.0 + 0.5 * GLASS_WALL_WIDTH_MM,
        )
        panel_axis.grid(True, which="major", alpha=0.22)
        panel_axis.grid(True, which="minor", alpha=0.08)
        panel_axis.minorticks_on()

    offset_label, _ = _offset_text(offset_mm)
    figure.suptitle(
        "Same-Axis Force-Gradient Comparison",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.968,
        f"{offset_label}; stationary atom; gravity excluded; negative is locally restoring",
        ha="center",
        va="top",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.944,
        "2-level pMOT cooling: P = 1.0571 mW per traveling beam "
        r"($s_0=I_0/I_{sat}=1.000$); trapping-path powers fixed at "
        "(19.147243, 19.147243, 38.294486) mW",
        ha="center",
        va="top",
        fontsize=11.5,
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MODEL_COLORS[model],
                linestyle=MODEL_STYLES[model],
                linewidth=2.2,
                label=MODEL_LABELS[model],
            )
            for model in MODEL_LABELS
        ]
        + [
            Patch(
                facecolor=GLASS_WALL_COLOR,
                alpha=0.20,
                label="glass-cell wall regions (center +/-10 mm, width 5 mm)",
            )
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.006),
        frameon=False,
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.98, 0.905), h_pad=2.0, w_pad=1.8)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _central_summary(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODEL_LABELS:
        for plane, scan_axis, transverse_axis in PANEL_SPECS:
            for offset_mm in OFFSETS_MM:
                subset = _panel_data(
                    frame,
                    model=model,
                    plane=plane,
                    scan_axis=scan_axis,
                    transverse_axis=transverse_axis,
                    offset_mm=offset_mm,
                )
                origin = subset.iloc[
                    int(np.argmin(np.abs(subset["scan_coordinate_mm"].to_numpy())))
                ]
                rows.append(
                    {
                        "model": model,
                        "plane": plane,
                        "scan_axis": scan_axis,
                        "transverse_axis": transverse_axis,
                        "transverse_offset_mm": offset_mm,
                        "force_at_scan_origin_zN": float(origin["same_axis_force_zN"]),
                        "force_gradient_at_scan_origin_1e19_n_per_m": float(
                            origin["same_axis_force_gradient_1e19_n_per_m"]
                        ),
                    }
                )
    return rows


def run(output_directory: Path, *, reuse_pmot_data: bool = False) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    figures = output_directory / "figures"
    figures.mkdir(exist_ok=True)
    pmot_csv = output_directory / "pMOT_2Level_linecuts.csv"
    scan_m = _scan_coordinates_m()

    multilevel = _load_multilevel_reference()
    mot_strange = _load_mot_strange_reference()
    if reuse_pmot_data and pmot_csv.is_file():
        pmot = pd.read_csv(pmot_csv)
        print(f"[pMOT_2Level] reusing {pmot_csv}", flush=True)
    else:
        pmot = _evaluate_pmot_2level(scan_m)
        pmot.to_csv(pmot_csv, index=False)

    columns = (
        "model",
        "model_label",
        "plane",
        "scan_axis",
        "transverse_axis",
        "transverse_offset_mm",
        "scan_coordinate_mm",
        "same_axis_force_n",
        "same_axis_force_zN",
        "same_axis_force_gradient_n_per_m",
        "same_axis_force_gradient_1e19_n_per_m",
    )
    frame = pd.concat(
        (multilevel[list(columns)], mot_strange[list(columns)], pmot[list(columns)]),
        ignore_index=True,
    )
    combined_csv = output_directory / "three_model_force_gradient_linecuts.csv"
    frame.to_csv(combined_csv, index=False)

    output_paths: dict[str, list[str]] = {
        "mot_strange_force_and_gradient": [],
        "pMOT_2Level_force_and_gradient": [],
        "three_model_gradient_overlay": [],
    }
    for offset_mm in OFFSETS_MM:
        _, suffix = _offset_text(offset_mm)
        for model, collection in (
            ("mot_strange", "mot_strange_force_and_gradient"),
            ("pMOT_2Level", "pMOT_2Level_force_and_gradient"),
        ):
            path = figures / f"{model}_force_gradient_{suffix}.png"
            _plot_force_and_gradient(
                frame,
                model=model,
                offset_mm=offset_mm,
                output_path=path,
            )
            output_paths[collection].append(str(path.relative_to(PROJECT_ROOT)))
        overlay_path = figures / f"three_model_force_gradient_overlay_{suffix}.png"
        _plot_gradient_overlay(frame, offset_mm=offset_mm, output_path=overlay_path)
        output_paths["three_model_gradient_overlay"].append(
            str(overlay_path.relative_to(PROJECT_ROOT))
        )

    simple = replace(default_simple_mot_config(), include_gravity=False)
    manifest = {
        "schema": "pmot.three-model-force-gradient-review.v1",
        "scientific_status": (
            "diagnostic comparison; pMOT_2Level is provisional and is not a "
            "validated physical pMOT prediction"
        ),
        "sampling": {
            "stationary_atom": True,
            "gravity_included": False,
            "scan_range_mm": [-SCAN_LIMIT_MM, SCAN_LIMIT_MM],
            "samples_per_cut": len(scan_m),
            "transverse_offsets_mm": list(OFFSETS_MM),
            "force_gradient_definition": (
                "numpy.gradient of the displayed same-axis force versus the "
                "saved nonuniform coordinate, edge_order=2"
            ),
        },
        "models": {
            "mot_multilevel": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "axial_coil_gradient_g_per_cm": 20.0,
                "source_csv": str(MULTILEVEL_CSV.relative_to(PROJECT_ROOT)),
            },
            "mot_strange": {
                "solver": "replacement 24-state Section-12 population-rate solver",
                "field": "centered intensity-derived surrogate magnetic field",
                "source_csv": str(MOT_STRANGE_CSV.relative_to(PROJECT_ROOT)),
            },
            "pMOT_2Level": {
                "solver": "provisional effective F=2 to F'=3 mean-force closure",
                "simple_configuration": asdict(simple),
                "cooling_power_mw_per_beam": PMOT_COOLING_POWER_MW_PER_BEAM,
                "raw_peak_saturation_parameter_per_beam": (
                    2.0
                    * PMOT_COOLING_POWER_MW_PER_BEAM
                    * 1.0e-3
                    / (
                        np.pi
                        * default_simple_mot_apparatus().cooling.beam_radius_m**2
                        * simple.saturation_intensity_w_per_m2
                    )
                ),
                "cooling_detuning_mhz": -15.0,
                "rate_normalization": (
                    "ordinary-frequency linewidth gamma_nu=Gamma/(2*pi); "
                    "scattering prefactor Gamma/2=pi*gamma_nu"
                ),
                "trapping_path_powers_mw": [
                    1.0e3 * value for value in TRAPPING_PATH_POWERS_W
                ],
                "trapping_helicities_xyz": ["sigma+", "sigma+", "sigma-"],
                "external_magnetic_field": "zero",
            },
        },
        "outputs": {
            "combined_linecuts_csv": str(combined_csv.relative_to(PROJECT_ROOT)),
            "pMOT_2Level_linecuts_csv": str(pmot_csv.relative_to(PROJECT_ROOT)),
            **output_paths,
        },
        "central_summary": _central_summary(frame),
        "interpretation": (
            "Negative same-axis force gradient is locally restoring. These "
            "one-dimensional cuts do not replace the full 3D force Jacobian."
        ),
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_directory": str(output_directory),
        "manifest": str(manifest_path),
        "figures": output_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-pmot-data",
        action="store_true",
        help="regenerate plots from an existing pMOT_2Level_linecuts.csv",
    )
    args = parser.parse_args()
    result = run(args.output.resolve(), reuse_pmot_data=args.reuse_pmot_data)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
