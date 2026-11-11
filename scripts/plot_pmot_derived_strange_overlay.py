"""Compare pMOT_2Level with a mot_strange field derived from the same pMOT.

The effective magnetic field in this diagnostic is not read from the retained
``mot_strange.surrogate_effective_field`` implementation.  At every sampled
position it evaluates the exact same pMOT trapping-beam objects, powers, and
polarizability table used by ``pMOT_2Level`` and applies the stretched cycling
transition mapping

    B_eff = DeltaU_vector / (mu_B * G),  G = 1.

The resulting field is passed to the replacement 24-state population-rate MOT.
Cooling power, detuning, Cartesian geometry, and propagation-frame helicities
are matched between the branches. The selected two-level curves may retain
any scalar/vector/tensor subset. The surrogate reference always retains the
unmasked vector field. Scalar and tensor shifts cannot be represented by that
magnetic-field surrogate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import perf_counter
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pmot.magnetic_fields import default_anti_helmholtz_config
from pmot.mot_multilevel import build_multilevel_mot_beams
from pmot.mot_multilevel import build_rate_equation_model
from pmot.mot_multilevel import default_multilevel_mot_config
from pmot.mot_multilevel import rate_equation_observable
from pmot.mot_multilevel.atomic_structure import BOHR_MAGNETON_J_PER_T
from pmot.mot_simple.configuration import default_simple_mot_apparatus
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import evaluate_differential_ac_stark_shift
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table
from pmot.pMOT_2Level import load_pmot_2level_component_polarizability_table
from pmot.pMOT_2Level import normalize_pmot_stark_component_code
from pmot.pMOT_2Level import pmot_stark_component_model_label
from pmot.pMOT_2Level import get_pmot_2level_pathway
from pmot.pMOT_2Level import PMOT_2LEVEL_PATHWAYS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "model_comparisons"
    / "pmot_derived_strange_overlay_20261025"
)
DEFAULT_VECTOR_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "model_comparisons"
    / "pmot_vector_only_derived_strange_overlay_20261025"
)
DEFAULT_FOUR_MODEL_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "model_comparisons"
    / "pmot_matched_four_model_overlay_20261025"
)
AXES = "xyz"
AXIS_INDEX = {axis: index for index, axis in enumerate(AXES)}
TRANSITION_ZEEMAN_FACTOR = 1.0


def scan_coordinates_m(scan_limit_mm: float) -> np.ndarray:
    """Return a broad linecut with extra resolution at zero and both foci."""

    broad = np.linspace(-scan_limit_mm, scan_limit_mm, 401)
    center_extent = min(1.0, scan_limit_mm)
    center = np.linspace(-center_extent, center_extent, 161)
    focus_arrays = []
    for center_mm in (-10.0, 10.0):
        if abs(center_mm) <= scan_limit_mm + 0.5:
            focus_arrays.append(np.linspace(center_mm - 0.3, center_mm + 0.3, 241))
    values_mm = np.unique(
        np.round(np.concatenate((broad, center, *focus_arrays)), decimals=10)
    )
    return 1.0e-3 * values_mm


def pMOT_derived_effective_field_t(
    position_m,
    *,
    trapping_beams,
    trapping_powers_w,
    polarizability_table,
    previous_axis=(0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return the pMOT vector Stark shift represented as an effective B field."""

    sample = evaluate_differential_ac_stark_shift(
        trapping_beams,
        position_m,
        np.zeros(3),
        path_powers_w=trapping_powers_w,
        previous_axis=previous_axis,
        polarizability_table=polarizability_table,
    )
    return np.asarray(sample.net_vector_energy_vector_j, dtype=float) / (
        BOHR_MAGNETON_J_PER_T * TRANSITION_ZEEMAN_FACTOR
    )


def evaluate_linecuts(
    scan_m: np.ndarray,
    *,
    cooling_power_mw: float,
    cooling_detuning_mhz: float,
    repump_power_mw: float,
    trapping_powers_mw: np.ndarray,
    two_level_variant: str = "full",
    mot_axial_gradient_g_per_cm: float = 20.0,
    pmot_components: tuple[str, ...] | None = None,
    pathway_name: str = "scalar_tensor_cancellation",
) -> tuple[pd.DataFrame, dict]:
    """Evaluate both force laws at identical stationary-atom positions."""

    apparatus = default_simple_mot_apparatus()
    simple = replace(
        default_simple_mot_config(),
        cooling_detuning_hz=1.0e6 * cooling_detuning_mhz,
        include_gravity=False,
    )
    apparatus = replace(
        apparatus,
        cooling=replace(
            apparatus.cooling,
            power_w_per_beam=1.0e-3 * cooling_power_mw,
            detuning_hz=1.0e6 * cooling_detuning_mhz,
        ),
    )
    cooling = build_pmot_2level_cooling_beams(
        apparatus_config=apparatus,
        simple_config=simple,
    )
    pathway = get_pmot_2level_pathway(pathway_name)
    trapping = pathway.build_trapping_beams()
    table = load_pmot_2level_polarizability_table()
    trapping_powers_w = 1.0e-3 * np.asarray(trapping_powers_mw, dtype=float)

    multilevel_config = replace(
        default_multilevel_mot_config(),
        cooling_power_w_per_beam=1.0e-3 * cooling_power_mw,
        repump_power_w_per_beam=1.0e-3 * repump_power_mw,
        cooling_detuning_rad_per_s=2.0 * np.pi * 1.0e6 * cooling_detuning_mhz,
        include_gravity=False,
    )
    multilevel_beams = build_multilevel_mot_beams(
        apparatus_config=apparatus,
        config=multilevel_config,
    )
    rate_model = build_rate_equation_model()
    fallback_axis = (0.0, 0.0, 1.0)
    velocity = np.zeros(3)

    def evaluate_two_level(position, selected_table):
        return evaluate_pmot_2level_local_observable(
            cooling,
            trapping,
            position,
            velocity,
            path_powers_w=trapping_powers_w,
            previous_axis=fallback_axis,
            simple_config=simple,
            polarizability_table=selected_table,
        )

    variants = {"full": ("svt",), "vector-only": ("v",), "both": ("svt", "v")}
    if two_level_variant not in variants:
        raise ValueError("two_level_variant must be 'full', 'vector-only', or 'both'")
    codes = tuple(normalize_pmot_stark_component_code(code) for code in (
        variants[two_level_variant] if pmot_components is None else pmot_components
    ))
    if not codes or len(set(codes)) != len(codes):
        raise ValueError("pMOT component selections must be nonempty and unique")
    two_level_evaluators = {
        pmot_stark_component_model_label(code): partial(
            evaluate_two_level,
            selected_table=load_pmot_2level_component_polarizability_table(code),
        )
        for code in codes
    }

    include_conventional_mot = two_level_variant == "both" or pmot_components is not None
    conventional_coil = default_anti_helmholtz_config(
        target_gradient_g_per_cm=mot_axial_gradient_g_per_cm
    )

    def derived_field(position_m):
        return pMOT_derived_effective_field_t(
            position_m,
            trapping_beams=trapping,
            trapping_powers_w=trapping_powers_w,
            polarizability_table=table,
            previous_axis=fallback_axis,
        )

    force = {
        **{
            model: {axis: np.empty(len(scan_m)) for axis in AXES}
            for model in two_level_evaluators
        },
        "pmot_derived_mot_strange": {
            axis: np.empty(len(scan_m)) for axis in AXES
        },
    }
    if include_conventional_mot:
        force["mot_multilevel"] = {
            axis: np.empty(len(scan_m)) for axis in AXES
        }
    start = perf_counter()
    for axis in AXES:
        component = AXIS_INDEX[axis]
        for index, coordinate_m in enumerate(scan_m):
            position = np.zeros(3)
            position[component] = coordinate_m
            two_level_samples = {
                model: evaluator(position)
                for model, evaluator in two_level_evaluators.items()
            }
            strange = rate_equation_observable(
                rate_model,
                multilevel_beams,
                tuple(position),
                tuple(velocity),
                None,
                multilevel_config,
                fallback_axis,
                magnetic_field_function=derived_field,
            )
            for model, sample in two_level_samples.items():
                force[model][axis][index] = sample.force_n[component]
            force["pmot_derived_mot_strange"][axis][index] = strange.force_n[
                component
            ]
            if include_conventional_mot:
                conventional = rate_equation_observable(
                    rate_model,
                    multilevel_beams,
                    tuple(position),
                    tuple(velocity),
                    conventional_coil,
                    multilevel_config,
                    fallback_axis,
                )
                force["mot_multilevel"][axis][index] = conventional.force_n[
                    component
                ]
        print(
            f"completed {axis}-axis ({len(scan_m)} points); "
            f"elapsed={perf_counter() - start:.1f} s",
            flush=True,
        )

    rows = []
    origin_index = int(np.argmin(np.abs(scan_m)))
    origin_gradients = {}
    for model, by_axis in force.items():
        origin_gradients[model] = {}
        for axis, values in by_axis.items():
            gradients = np.gradient(values, scan_m, edge_order=2)
            origin_gradients[model][axis] = float(gradients[origin_index])
            for coordinate_m, force_n, gradient_n_per_m in zip(
                scan_m, values, gradients, strict=True
            ):
                rows.append(
                    {
                        "model": model,
                        "axis": axis,
                        "coordinate_mm": 1.0e3 * coordinate_m,
                        "force_n": force_n,
                        "force_zN": 1.0e21 * force_n,
                        "force_gradient_n_per_m": gradient_n_per_m,
                        "force_gradient_1e19_n_per_m": 1.0e19
                        * gradient_n_per_m,
                    }
                )

    step_m = 1.0e-6
    field_jacobian = np.empty((3, 3))
    for coordinate in range(3):
        displacement = np.zeros(3)
        displacement[coordinate] = step_m
        field_jacobian[:, coordinate] = (
            derived_field(displacement) - derived_field(-displacement)
        ) / (2.0 * step_m)

    metadata = {
        "comparison": (
            f"{', '.join(two_level_evaluators)} versus replacement 24-state MOT using a field "
            "derived pointwise from the exact same pMOT vector Stark geometry"
        ),
        "two_level_variant": two_level_variant if pmot_components is None else "components",
        "two_level_model_keys": list(two_level_evaluators),
        "stark_components": [code or "none" for code in codes],
        "pathway": pathway.key,
        "trapping_wavelength_nm": pathway.trapping_wavelength_nm,
        "stationary_atom": True,
        "gravity_excluded": True,
        "cooling_power_mw_per_component_all_models": cooling_power_mw,
        "cooling_detuning_mhz_all_models": cooling_detuning_mhz,
        "cooling_helicities_xyz_both_directions_all_models": [
            "sigma+",
            "sigma+",
            "sigma-",
        ],
        "multilevel_repump_power_mw_per_component": repump_power_mw,
        "two_level_repump_role": "passive metadata; F=1 is absent from closure",
        "trapping_path_powers_mw_xyz": trapping_powers_mw.tolist(),
        "trapping_geometry_source": "pmot.pMOT_2Level.build_pmot_2level_trapping_beams",
        "effective_field_equation": "B_eff = DeltaU_vector/(mu_B*G)",
        "surrogate_reference_stark_components": "v (unmasked reference)",
        "transition_zeeman_factor_G": TRANSITION_ZEEMAN_FACTOR,
        "origin_effective_field_gradient_g_per_cm_xyz": (
            100.0 * np.diag(field_jacobian)
        ).tolist(),
        "conventional_mot_field_gradient_g_per_cm_xyz": (
            [
                0.5 * mot_axial_gradient_g_per_cm,
                0.5 * mot_axial_gradient_g_per_cm,
                -mot_axial_gradient_g_per_cm,
            ]
            if include_conventional_mot
            else None
        ),
        "origin_force_gradient_1e19_n_per_m": {
            model: {
                axis: 1.0e19 * value for axis, value in by_axis.items()
            }
            for model, by_axis in origin_gradients.items()
        },
        "remaining_intentional_differences": [
            "effective two-level saturated scattering closure versus 24-state population-rate solver",
            "multilevel repumper is dynamically active; two-level repumper is passive because no F=1 population exists",
        ]
        + (
            ["selected pMOTs retain the specified scalar/tensor ranks; field surrogate always uses the full vector shift only"]
            if any("s" in code or "t" in code for code in codes)
            else []
        ),
    }
    return pd.DataFrame(rows), metadata


def plot_overlay(data: pd.DataFrame, metadata: dict, output_path: Path) -> None:
    """Plot same-axis force above its numerical derivative for x, y, and z."""

    available_models = set(data["model"])
    all_labels = {
        "mot_multilevel": "24-state conventional MOT",
        "pmot_derived_mot_strange": "24-state mot_strange from this pMOT",
    }
    model_order = (
        "mot_multilevel",
        "pmot_derived_mot_strange",
    )
    labels = {
        model: all_labels[model]
        for model in model_order
        if model in available_models
    }
    colors = {
        "mot_multilevel": "#111827",
        "pmot_derived_mot_strange": "#d97706",
    }
    styles = {
        "mot_multilevel": "-.",
        "pmot_derived_mot_strange": "--",
    }
    palette = ("#be123c", "#2563eb", "#0f766e", "#7c3aed", "#0891b2", "#a16207", "#be185d", "#475569")
    for index, model in enumerate(metadata["two_level_model_keys"]):
        labels[model] = model
        colors[model] = palette[index % len(palette)]
        styles[model] = ("-", ":", "--", "-.")[index % 4]
    figure, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), sharex="col")
    for column, axis in enumerate(AXES):
        for model in labels:
            subset = data[(data["model"] == model) & (data["axis"] == axis)]
            axes[0, column].plot(
                subset["coordinate_mm"],
                subset["force_zN"],
                color=colors[model],
                linestyle=styles[model],
                linewidth=1.8,
                label=labels[model],
            )
            axes[1, column].plot(
                subset["coordinate_mm"],
                subset["force_gradient_1e19_n_per_m"],
                color=colors[model],
                linestyle=styles[model],
                linewidth=1.8,
            )
        fixed_coordinates = {"x": "y = z = 0", "y": "x = z = 0", "z": "x = y = 0"}
        axes[0, column].set_title(
            f"Along {axis}-axis; {fixed_coordinates[axis]}"
        )
        axes[0, column].set_ylabel(rf"$F_{axis}$ [zN]")
        axes[1, column].set_ylabel(
            rf"$\partial F_{axis}/\partial {axis}$ [$10^{{-19}}$ N/m]"
        )
        axes[1, column].set_xlabel(f"{axis} position [mm]")
        for panel in axes[:, column]:
            panel.axhline(0.0, color="0.25", linewidth=0.8)
            panel.axvline(0.0, color="0.25", linewidth=0.8)
            for wall_center in (-10.0, 10.0):
                panel.axvspan(
                    wall_center - 2.5,
                    wall_center + 2.5,
                    color="#94a3b8",
                    alpha=0.12,
                )
            panel.set_xlim(data["coordinate_mm"].min(), data["coordinate_mm"].max())
            panel.grid(alpha=0.20)
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    powers = metadata["trapping_path_powers_mw_xyz"]
    figure.suptitle(
        "Selected 2-Level pMOTs and pMOT-Derived mot_strange: Force and Force Gradient",
        fontsize=16.5,
        y=0.995,
    )
    conventional_clause = (
        "conventional MOT gradient "
        f"({metadata['conventional_mot_field_gradient_g_per_cm_xyz'][0]:+.5g}, "
        f"{metadata['conventional_mot_field_gradient_g_per_cm_xyz'][1]:+.5g}, "
        f"{metadata['conventional_mot_field_gradient_g_per_cm_xyz'][2]:+.5g}) G/cm; "
        if metadata["conventional_mot_field_gradient_g_per_cm_xyz"] is not None
        else ""
    )
    figure.text(
        0.5,
        0.958,
        fill((
            f"Same pMOT trapping geometry and powers ({powers[0]:.6g}, "
            f"{powers[1]:.6g}, {powers[2]:.6g}) mW/path; same cooling "
            f"{metadata['cooling_power_mw_per_component_all_models']:.6g} "
            f"mW/beam at {metadata['cooling_detuning_mhz_all_models']:.6g} MHz; "
            f"{conventional_clause}"
            f"λ = {metadata['trapping_wavelength_nm']:.9f} nm; "
            "stationary atom; radiation pressure; gravity excluded"
        ), width=160),
        ha="center",
        va="top",
        fontsize=10.5,
    )
    figure.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=min(len(legend_labels), 4),
        frameon=False,
    )
    legend_bottom = 0.045 + 0.028 * int(np.ceil(len(legend_labels) / 4))
    figure.tight_layout(rect=(0.02, legend_bottom, 0.98, 0.87), h_pad=1.4, w_pad=1.3)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--two-level-variant",
        choices=("full", "vector-only", "both"),
        default="full",
    )
    parser.add_argument("--scan-limit-mm", type=float, default=12.5)
    parser.add_argument("--cooling-power-mw", type=float, default=1.0571)
    parser.add_argument("--cooling-detuning-mhz", type=float, default=-15.0)
    parser.add_argument("--repump-power-mw", type=float, default=0.1)
    parser.add_argument(
        "--pmot-components", nargs="+", default=None,
        help="Stark selections to overlay, e.g. svt sv v none; overrides --two-level-variant.",
    )
    parser.add_argument("--pathway", choices=tuple(PMOT_2LEVEL_PATHWAYS), default="scalar_tensor_cancellation")
    parser.add_argument(
        "--mot-axial-gradient-g-per-cm",
        type=float,
        default=20.0,
        help="positive anti-Helmholtz axial-gradient magnitude; Bz gradient is negative",
    )
    parser.add_argument(
        "--trapping-powers-mw",
        type=float,
        nargs=3,
        metavar=("PX", "PY", "PZ"),
        default=(31.524310589374092,) * 3,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scan_limit_mm <= 0.0:
        raise ValueError("scan-limit-mm must be positive")
    if args.mot_axial_gradient_g_per_cm <= 0.0:
        raise ValueError("mot-axial-gradient-g-per-cm must be positive")
    trapping_powers_mw = np.asarray(args.trapping_powers_mw, dtype=float)
    if np.any(trapping_powers_mw < 0.0) or not np.all(np.isfinite(trapping_powers_mw)):
        raise ValueError("trapping powers must be finite and non-negative")
    default_output_directory = {
        "full": DEFAULT_OUTPUT_DIRECTORY,
        "vector-only": DEFAULT_VECTOR_OUTPUT_DIRECTORY,
        "both": DEFAULT_FOUR_MODEL_OUTPUT_DIRECTORY,
    }[args.two_level_variant]
    if args.pmot_components is not None:
        tag = "_vs_".join(normalize_pmot_stark_component_code(code) or "none" for code in args.pmot_components)
        default_output_directory = PROJECT_ROOT / "outputs" / "figures" / "model_comparisons" / f"pmot_derived_{args.pathway}_{tag}"
    output_directory = (
        default_output_directory
        if args.output_directory is None
        else args.output_directory
    ).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    scan_m = scan_coordinates_m(args.scan_limit_mm)
    data, metadata = evaluate_linecuts(
        scan_m,
        cooling_power_mw=args.cooling_power_mw,
        cooling_detuning_mhz=args.cooling_detuning_mhz,
        repump_power_mw=args.repump_power_mw,
        trapping_powers_mw=trapping_powers_mw,
        two_level_variant=args.two_level_variant,
        mot_axial_gradient_g_per_cm=args.mot_axial_gradient_g_per_cm,
        pmot_components=args.pmot_components,
        pathway_name=args.pathway,
    )
    csv_path = output_directory / "force_gradient_linecuts.csv"
    figure_stem = {
        "full": "pmot_derived_strange_force_gradient_overlay",
        "vector-only": "pmot_vector_only_derived_strange_force_gradient_overlay",
        "both": "pmot_matched_four_model_force_gradient_overlay",
    }[args.two_level_variant]
    if args.pmot_components is not None:
        figure_stem = "pmot_" + "_vs_".join(metadata["stark_components"]) + "_derived_strange_overlay"
    figure_path = output_directory / f"{figure_stem}.png"
    metadata_path = output_directory / "metadata.json"
    data.to_csv(csv_path, index=False)
    metadata.update(
        {
            "scan_limit_mm": args.scan_limit_mm,
            "sample_count_per_axis": len(scan_m),
            "data_csv": str(csv_path),
            "figure": str(figure_path),
        }
    )
    plot_overlay(data, metadata, figure_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
