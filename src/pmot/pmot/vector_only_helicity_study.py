"""Damping-preserving vector-only trapping-helicity audit.

This diagnostic is intentionally separate from the full provisional
scalar/vector/tensor Stark study.  It imposes the intended magic-wavelength
idealization by applying only the vector differential transition shift to the
unchanged 24-state cooling/repump rate kernel.  The authoritative 780-nm beam
helicities remain fixed while all 64 independent sigma-only trapping-light
helicity labels are enumerated.

Only the eight incident=retro choices have zero vector bias at the origin and
are therefore assigned local equilibrium classifications.  The reported
position response is a 1529-nm-vector-induced change in 780-nm scattering; no
direct 1529-nm radiation-pressure or conservative force is included.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from ..configuration import PLANCK_CONSTANT_J_S
from .ac_stark import EFFECTIVE_DETUNING_EQUATION
from .ac_stark import PROVISIONAL_MODEL_NAME
from .configuration import pmot_paths
from .helicity_sweep import CURRENT_COMBINED_CODE
from .helicity_sweep import CURRENT_PATH_CODE
from .helicity_sweep import HELICITY_CODE_ORDER
from .helicity_sweep import INHERITED_ABSORPTION_FORCE_CAVEAT
from .helicity_sweep import HelicitySweepContext
from .helicity_sweep import all_helicity_codes
from .helicity_sweep import build_helicity_sweep_context
from .helicity_sweep import classify_position_jacobian
from .helicity_sweep import classify_velocity_jacobian
from .helicity_sweep import decode_helicity_code
from .helicity_sweep import path_helicity_codes
from .vector_only import BARE_AXIS
from .vector_only import VectorOnlyObservable
from .vector_only import bare_780_observable
from .vector_only import prepare_vector_only_trapping_environment
from .vector_only import vector_only_pmot_observable


VECTOR_ONLY_MODEL_NAME = "ideal_magic_scalar_tensor_cancelled_vector_transition_proxy"
VECTOR_ONLY_SHIFT_EQUATION = (
    "Delta_omega_applied[g->e] = DeltaE_vector[g->e] / hbar; "
    "DeltaE_scalar = DeltaE_tensor = 0 by imposed magic-wavelength idealization"
)
FORCE_MECHANISM = (
    "F_proxy = sum_b hbar*k_780,b*R_abs,b evaluated after the 1529-nm vector "
    "transition-resonance shift; no direct 1529-nm force is applied"
)
REVERSED_PATH_CODE = "--+"
REVERSED_COMBINED_CODE = REVERSED_PATH_CODE + REVERSED_PATH_CODE


def _cycling_transition_index(model) -> int:
    for index, transition in enumerate(model.structure.absorption_transitions):
        if (
            transition.ground_f,
            transition.ground_m_f,
            transition.excited_f,
            transition.excited_m_f,
        ) == (2, 2, 3, 3):
            return index
    raise RuntimeError("the rate model is missing the stretched cycling transition")


def _centered_jacobian(
    vector_function: Callable[[np.ndarray], np.ndarray],
    step: float,
) -> np.ndarray:
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be finite and positive")
    output = np.empty((3, 3), dtype=float)
    for column in range(3):
        offset = np.zeros(3)
        offset[column] = step
        output[:, column] = (
            np.asarray(vector_function(offset), dtype=float)
            - np.asarray(vector_function(-offset), dtype=float)
        ) / (2.0 * step)
    return output


def bare_780_jacobians(
    context: HelicitySweepContext,
    *,
    position_step_m: float = 2.0e-6,
    velocity_step_m_per_s: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return no-Stark position and velocity controls at the origin."""

    zero = np.zeros(3)
    position = _centered_jacobian(
        lambda offset: bare_780_observable(context, offset, zero).force_n,
        position_step_m,
    )
    velocity = _centered_jacobian(
        lambda offset: bare_780_observable(context, zero, offset).force_n,
        velocity_step_m_per_s,
    )
    return position, velocity


def _complex_list(values) -> list[dict[str, float]]:
    return [
        {"real": float(np.real(value)), "imaginary": float(np.imag(value))}
        for value in values
    ]


def evaluate_vector_only_configuration(
    context: HelicitySweepContext,
    code: str,
    *,
    position_step_m: float = 2.0e-6,
    velocity_step_m_per_s: float = 1.0e-3,
    field_zero_tolerance_g: float = 1.0e-6,
    force_equilibrium_tolerance_n: float = 1.0e-28,
) -> dict[str, Any]:
    """Record every code, but linearize only a centered origin equilibrium."""

    prepared = prepare_vector_only_trapping_environment(context, code)

    def evaluate(position, velocity):
        return vector_only_pmot_observable(
            context,
            code,
            position,
            velocity,
            prepared=prepared,
        )

    zero = np.zeros(3)
    origin = evaluate(zero, zero)
    force = np.asarray(origin.rate_equation.force_n, dtype=float)
    field_t = np.asarray(origin.stark_diagnostic.effective_field_t, dtype=float)
    incident_code = code[:3]
    retro_code = code[3:]
    pair_centered = incident_code == retro_code
    field_zero = float(np.linalg.norm(field_t) * 1.0e4) <= field_zero_tolerance_g
    force_equilibrium = (
        float(np.linalg.norm(force)) <= force_equilibrium_tolerance_n
    )
    centered_equilibrium = pair_centered and field_zero and force_equilibrium
    record: dict[str, Any] = {
        "combined_code": code,
        "incident_code_xyz": incident_code,
        "retro_code_xyz": retro_code,
        "mismatched_path_count": sum(
            left != right for left, right in zip(incident_code, retro_code)
        ),
        "pair_centered": pair_centered,
        "origin_effective_field_zero": field_zero,
        "origin_force_equilibrium": force_equilibrium,
        "centered_origin_equilibrium": centered_equilibrium,
        "origin_force_n": force.tolist(),
        "origin_force_norm_n": float(np.linalg.norm(force)),
        "origin_effective_field_proxy_t": field_t.tolist(),
        "origin_effective_field_proxy_magnitude_g": float(
            np.linalg.norm(field_t) * 1.0e4
        ),
        "position_jacobian_n_per_m": None,
        "velocity_jacobian_n_s_per_m": None,
        "position_jacobian_eigenvalues_n_per_m": None,
        "velocity_jacobian_eigenvalues_n_s_per_m": None,
        "position_classification": "not_classified_origin_biased",
        "velocity_classification": "not_classified_origin_biased",
        "combined_classification": "not_classified_origin_biased",
    }
    if not centered_equilibrium:
        return record

    position_jacobian = _centered_jacobian(
        lambda offset: evaluate(offset, zero).rate_equation.force_n,
        position_step_m,
    )
    velocity_jacobian = _centered_jacobian(
        lambda offset: evaluate(zero, offset).rate_equation.force_n,
        velocity_step_m_per_s,
    )
    position_classification = classify_position_jacobian(position_jacobian)
    velocity_classification = classify_velocity_jacobian(velocity_jacobian)
    if position_classification == "restoring" and velocity_classification == "damping":
        combined = "restoring_and_damping"
    elif position_classification == "anti_restoring" and velocity_classification == "damping":
        combined = "anti_restoring_but_damping"
    else:
        combined = f"{position_classification}_and_{velocity_classification}"
    record.update(
        {
            "position_jacobian_n_per_m": position_jacobian.tolist(),
            "velocity_jacobian_n_s_per_m": velocity_jacobian.tolist(),
            "position_jacobian_eigenvalues_n_per_m": _complex_list(
                np.linalg.eigvals(position_jacobian)
            ),
            "velocity_jacobian_eigenvalues_n_s_per_m": _complex_list(
                np.linalg.eigvals(velocity_jacobian)
            ),
            "position_classification": position_classification,
            "velocity_classification": velocity_classification,
            "combined_classification": combined,
        }
    )
    return record


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "origin_force_n",
            "origin_effective_field_proxy_t",
            "position_jacobian_n_per_m",
            "velocity_jacobian_n_s_per_m",
            "position_jacobian_eigenvalues_n_per_m",
            "velocity_jacobian_eigenvalues_n_s_per_m",
        }
    }
    for index, axis in enumerate("xyz"):
        row[f"origin_force_{axis}_n"] = record["origin_force_n"][index]
        row[f"origin_effective_field_proxy_{axis}_t"] = record[
            "origin_effective_field_proxy_t"
        ][index]
    for force_index, force_axis in enumerate("xyz"):
        for variable_index, variable_axis in enumerate("xyz"):
            position = record["position_jacobian_n_per_m"]
            velocity = record["velocity_jacobian_n_s_per_m"]
            row[f"dF{force_axis}_d{variable_axis}_n_per_m"] = (
                None if position is None else position[force_index][variable_index]
            )
            row[f"dF{force_axis}_dv{variable_axis}_n_s_per_m"] = (
                None if velocity is None else velocity[force_index][variable_index]
            )
    return row


def _counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    centered = [record for record in records if record["centered_origin_equilibrium"]]

    def count(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for record in centered:
            value = str(record[field])
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "enumerated": len(records),
        "centered_and_classified": len(centered),
        "origin_biased_not_classified": len(records) - len(centered),
        "mismatched_path_count": {
            str(value): sum(record["mismatched_path_count"] == value for record in records)
            for value in range(4)
        },
        "centered_position": count("position_classification"),
        "centered_velocity": count("velocity_classification"),
        "centered_combined": count("combined_classification"),
    }


def sample_vector_only_comparison(
    context: HelicitySweepContext,
    *,
    position_extent_m: float = 0.5e-3,
    velocity_extent_m_per_s: float = 0.5,
    sample_count: int = 81,
) -> pd.DataFrame:
    """Sample current/reversed vector-only and bare-780 control lineouts."""

    if sample_count < 3 or sample_count % 2 == 0:
        raise ValueError("sample_count must be an odd integer of at least three")
    reference = _cycling_transition_index(context.model)
    rows: list[dict[str, Any]] = []
    roles = (
        ("current_matched", CURRENT_COMBINED_CODE),
        ("reversed_matched", REVERSED_COMBINED_CODE),
    )
    for role, code in roles:
        prepared = prepare_vector_only_trapping_environment(context, code)
        for component, axis in enumerate("xyz"):
            for coordinate in np.linspace(-position_extent_m, position_extent_m, sample_count):
                position = np.zeros(3)
                position[component] = coordinate
                observable = vector_only_pmot_observable(
                    context,
                    code,
                    position,
                    np.zeros(3),
                    prepared=prepared,
                )
                stark = observable.stark_diagnostic
                rows.append(
                    {
                        "scan_type": "position",
                        "configuration_role": role,
                        "combined_code": code,
                        "axis": axis,
                        "coordinate_m": float(coordinate),
                        "velocity_m_per_s": np.nan,
                        "same_axis_780_force_proxy_n": float(
                            observable.rate_equation.force_n[component]
                        ),
                        "applied_reference_vector_transition_shift_hz": float(
                            stark.vector_transition_energy_j[reference]
                            / PLANCK_CONSTANT_J_S
                        ),
                        "same_axis_effective_field_proxy_g": float(
                            1.0e4 * stark.effective_field_t[component]
                        ),
                    }
                )
            for speed in np.linspace(
                -velocity_extent_m_per_s,
                velocity_extent_m_per_s,
                sample_count,
            ):
                velocity = np.zeros(3)
                velocity[component] = speed
                combined = vector_only_pmot_observable(
                    context,
                    code,
                    np.zeros(3),
                    velocity,
                    prepared=prepared,
                )
                bare = bare_780_observable(context, np.zeros(3), velocity)
                rows.extend(
                    (
                        {
                            "scan_type": "velocity_combined",
                            "configuration_role": role,
                            "combined_code": code,
                            "axis": axis,
                            "coordinate_m": np.nan,
                            "velocity_m_per_s": float(speed),
                            "same_axis_780_force_proxy_n": float(
                                combined.rate_equation.force_n[component]
                            ),
                            "applied_reference_vector_transition_shift_hz": np.nan,
                            "same_axis_effective_field_proxy_g": np.nan,
                        },
                        {
                            "scan_type": "velocity_bare_780",
                            "configuration_role": role,
                            "combined_code": code,
                            "axis": axis,
                            "coordinate_m": np.nan,
                            "velocity_m_per_s": float(speed),
                            "same_axis_780_force_proxy_n": float(bare.force_n[component]),
                            "applied_reference_vector_transition_shift_hz": 0.0,
                            "same_axis_effective_field_proxy_g": 0.0,
                        },
                    )
                )
    return pd.DataFrame(rows)


def _plot_all64(records: list[dict[str, Any]], path: Path) -> Path:
    codes = path_helicity_codes()
    lookup = {
        (record["incident_code_xyz"], record["retro_code_xyz"]): record
        for record in records
    }
    bias = np.asarray(
        [
            [lookup[(incident, retro)]["origin_effective_field_proxy_magnitude_g"] for retro in codes]
            for incident in codes
        ]
    )
    force = np.asarray(
        [
            [1.0e22 * lookup[(incident, retro)]["origin_force_norm_n"] for retro in codes]
            for incident in codes
        ]
    )
    figure, panels = plt.subplots(1, 2, figsize=(15.5, 7.0), constrained_layout=True)
    images = (
        panels[0].imshow(bias, origin="upper", cmap="viridis"),
        panels[1].imshow(force, origin="upper", cmap="magma"),
    )
    panels[0].set_title(r"Origin vector $B_{\rm eff}$ proxy magnitude [G]")
    panels[1].set_title(r"Origin 780-nm force-proxy magnitude [$10^{-22}$ N]")
    for panel_index, panel in enumerate(panels):
        panel.set_xticks(range(8), codes, rotation=45, ha="right")
        panel.set_yticks(range(8), codes)
        panel.set_xlabel("Retro trapping helicities (x y z)")
        panel.set_ylabel("Incident trapping helicities (x y z)")
        for row, incident in enumerate(codes):
            for column, retro in enumerate(codes):
                record = lookup[(incident, retro)]
                if not record["centered_origin_equilibrium"]:
                    label = "B"
                else:
                    label = {
                        "restoring_and_damping": "RD",
                        "anti_restoring_but_damping": "AD",
                    }.get(record["combined_classification"], "SD")
                value = bias[row, column] if panel_index == 0 else force[row, column]
                threshold = 12.0 if panel_index == 0 else 6.8
                panel.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color="black" if value >= threshold else "white",
                )
        for code, color in (
            (CURRENT_PATH_CODE, "#22d3ee"),
            (REVERSED_PATH_CODE, "#facc15"),
        ):
            index = codes.index(code)
            panel.add_patch(
                Rectangle(
                    (index - 0.48, index - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor=color,
                    linewidth=2.8,
                )
            )
    figure.colorbar(images[0], ax=panels[0], shrink=0.82)
    figure.colorbar(images[1], ax=panels[1], shrink=0.82)
    figure.suptitle(
        "Vector-only ideal-magic audit: all 64 trapping-helicity labels enumerated\n"
        "Only 8 diagonal zero-bias origins classified: RD = restoring+damping, "
        "AD = anti-restoring+damping, SD = saddle+damping, B = biased"
    )
    figure.text(
        0.5,
        -0.04,
        "Fixed authoritative 780-nm helicities; scalar and tensor shifts imposed zero",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_centered_jacobians(
    records: list[dict[str, Any]],
    bare_velocity_jacobian: np.ndarray,
    path: Path,
) -> Path:
    centered = [record for record in records if record["centered_origin_equilibrium"]]
    centered.sort(key=lambda record: path_helicity_codes().index(record["incident_code_xyz"]))
    labels = [record["incident_code_xyz"] for record in centered]
    position = np.asarray(
        [np.diag(record["position_jacobian_n_per_m"]) for record in centered]
    )
    velocity = np.asarray(
        [np.diag(record["velocity_jacobian_n_s_per_m"]) for record in centered]
    )
    figure, panels = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    x = np.arange(8)
    colors = ("#b91c1c", "#1d4ed8", "#15803d")
    markers = ("o", "s", "^")
    for component, (axis, color, marker) in enumerate(zip("xyz", colors, markers)):
        panels[0].plot(
            x,
            1.0e19 * position[:, component],
            color=color,
            marker=marker,
            label=rf"$\partial F_{axis}/\partial {axis}$",
        )
        panels[1].plot(
            x,
            1.0e22 * velocity[:, component],
            color=color,
            marker=marker,
            label=rf"combined $\partial F_{axis}/\partial v_{axis}$",
        )
        panels[1].axhline(
            1.0e22 * bare_velocity_jacobian[component, component],
            color=color,
            linestyle="--",
            alpha=0.45,
            linewidth=1.1,
        )
    panels[0].set(
        title="1529-vector-induced position response through 780 scattering",
        ylabel=r"Position slope [$10^{-19}$ N/m]",
    )
    panels[1].set(
        title="Combined velocity response; dashed = bare 780 controls",
        ylabel=r"Velocity slope [$10^{-22}$ N s/m]",
    )
    for panel in panels:
        panel.set_xticks(x, labels)
        panel.set_xlabel("Matched trapping helicities: incident = retro (x y z)")
        panel.axhline(0.0, color="0.35", linewidth=0.8)
        panel.grid(alpha=0.23)
        panel.legend(ncol=3, fontsize=8)
        for code, color in (
            (CURRENT_PATH_CODE, "#0891b2"),
            (REVERSED_PATH_CODE, "#ca8a04"),
        ):
            index = labels.index(code)
            panel.axvspan(index - 0.36, index + 0.36, color=color, alpha=0.10)
    panels[0].text(
        0.02,
        0.04,
        "negative = restoring; current (++-) is the unique all-axis choice",
        transform=panels[0].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )
    panels[1].text(
        0.02,
        0.04,
        "negative = damping; fixed red-detuned 780-nm cooling remains damping",
        transform=panels[1].transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.8"},
    )
    figure.suptitle(
        "Centered vector-only trapping-helicity audit at the 20-G/cm proxy scale\n"
        "No direct 1529-nm force; full 3x3 matrices and eigenvalues are in CSV/JSON"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_current_vs_reversed(dataframe: pd.DataFrame, path: Path) -> Path:
    roles = (
        ("current_matched", "Current matched (++-): restoring + damping"),
        ("reversed_matched", "Reversed matched (--+): anti-restoring + damping"),
    )
    colors = {"x": "#b91c1c", "y": "#1d4ed8", "z": "#15803d"}
    styles = {"x": "-", "y": "--", "z": ":"}
    figure, panels = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    for column, (role, title) in enumerate(roles):
        role_data = dataframe[dataframe["configuration_role"] == role]
        spatial = role_data[role_data["scan_type"] == "position"]
        combined_velocity = role_data[role_data["scan_type"] == "velocity_combined"]
        bare_velocity = role_data[role_data["scan_type"] == "velocity_bare_780"]
        for axis in "xyz":
            axis_space = spatial[spatial["axis"] == axis]
            coordinate_mm = 1.0e3 * axis_space["coordinate_m"].to_numpy()
            panels[0, column].plot(
                coordinate_mm,
                1.0e21 * axis_space["same_axis_780_force_proxy_n"],
                color=colors[axis],
                linestyle=styles[axis],
                linewidth=1.9,
                label=axis,
            )
            panels[1, column].plot(
                coordinate_mm,
                1.0e-6 * axis_space["applied_reference_vector_transition_shift_hz"],
                color=colors[axis],
                linestyle=styles[axis],
                linewidth=1.9,
                label=axis,
            )
            panels[2, column].plot(
                coordinate_mm,
                axis_space["same_axis_effective_field_proxy_g"],
                color=colors[axis],
                linestyle=styles[axis],
                linewidth=1.9,
                label=axis,
            )
            combined_axis = combined_velocity[combined_velocity["axis"] == axis]
            bare_axis = bare_velocity[bare_velocity["axis"] == axis]
            panels[3, column].plot(
                combined_axis["velocity_m_per_s"],
                1.0e21 * combined_axis["same_axis_780_force_proxy_n"],
                color=colors[axis],
                linestyle=styles[axis],
                linewidth=1.9,
                label=f"{axis}: combined",
            )
            panels[3, column].plot(
                bare_axis["velocity_m_per_s"],
                1.0e21 * bare_axis["same_axis_780_force_proxy_n"],
                color=colors[axis],
                linestyle="--",
                alpha=0.35,
                linewidth=1.1,
            )
        panels[0, column].set_title(title)
        panels[0, column].set_ylabel(r"780-nm force proxy [$10^{-21}$ N]")
        panels[1, column].set_ylabel("Applied vector transition shift [MHz]")
        panels[2, column].set_ylabel(r"Signed vector $B_{\rm eff}$ proxy [G]")
        panels[3, column].set_ylabel(r"780-nm force proxy [$10^{-21}$ N]")
        panels[3, column].set_xlabel("Same-axis velocity [m/s]")
        for row in range(3):
            panels[row, column].set_xlabel("Axis coordinate [mm]")
        for row in range(4):
            panels[row, column].axhline(0.0, color="0.4", linewidth=0.8)
            panels[row, column].axvline(0.0, color="0.4", linewidth=0.8)
            panels[row, column].grid(alpha=0.23)
            panels[row, column].legend(ncol=3, fontsize=8, loc="best")
    panels[0, 0].text(
        0.02,
        0.04,
        "stationary atom; response mediated only through 780-nm scattering",
        transform=panels[0, 0].transAxes,
        fontsize=9,
    )
    panels[1, 0].text(
        0.02,
        0.04,
        "stretched cycling-transition resonance shift; scalar=tensor=0 imposed",
        transform=panels[1, 0].transAxes,
        fontsize=9,
    )
    panels[2, 0].text(
        0.02,
        0.04,
        "optical-spin diagnostic; not an applied magnetic field",
        transform=panels[2, 0].transAxes,
        fontsize=9,
    )
    panels[3, 0].text(
        0.02,
        0.04,
        "colored = combined; faint dashed = bare 780-nm control",
        transform=panels[3, 0].transAxes,
        fontsize=9,
    )
    figure.suptitle(
        "Ideal-magic vector-only audit: fixed 780-nm damping and trapping-helicity reversal\n"
        "Position force is 1529-vector-induced through 780 scattering—not direct 1529 force"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _beam_manifest(context: HelicitySweepContext) -> list[dict[str, Any]]:
    return [
        {
            "label": beam.label,
            "family": beam.family,
            "axis_name": beam.axis_name,
            "propagation_sense": beam.propagation_sense,
            "propagation_frame_helicity": beam.circular_polarization,
            "direction": list(beam.direction),
            "power_w": beam.power_w,
            "wavelength_m": beam.wavelength_m,
            "detuning_hz": beam.detuning_hz,
        }
        for beam in context.cooling_repump_beams
    ]


def run_vector_only_helicity_study(
    *,
    power_w_per_path: float | None = None,
    target_gradient_g_per_cm: float = 20.0,
    position_step_m: float = 2.0e-6,
    velocity_step_m_per_s: float = 1.0e-3,
    comparison_position_extent_m: float = 0.5e-3,
    comparison_velocity_extent_m_per_s: float = 0.5,
    comparison_sample_count: int = 81,
    output_tag: str = "vector_only_helicity_20Gpcm",
) -> dict[str, Any]:
    """Run the distinct ideal-magic, damping-preserving helicity audit."""

    started = perf_counter()
    context = build_helicity_sweep_context(
        power_w_per_path=power_w_per_path,
        target_gradient_g_per_cm=target_gradient_g_per_cm,
    )
    roots = pmot_paths()
    statistics_root = roots["outputs_statistics"] / output_tag
    figure_root = roots["outputs_figures"] / output_tag
    statistics_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    print(
        "[vector-only helicity] fixed authoritative cooling/repump beams; "
        "scalar=tensor=0 imposed; external B=0; gravity excluded",
        flush=True,
    )
    print(
        "[vector-only helicity] trapping power="
        f"{1e3*context.power_w_per_path:.6f} mW/path; enumerating 64, "
        "classifying only 8 zero-bias origins",
        flush=True,
    )
    bare_position, bare_velocity = bare_780_jacobians(
        context,
        position_step_m=position_step_m,
        velocity_step_m_per_s=velocity_step_m_per_s,
    )
    records: list[dict[str, Any]] = []
    for index, code in enumerate(all_helicity_codes(), start=1):
        records.append(
            evaluate_vector_only_configuration(
                context,
                code,
                position_step_m=position_step_m,
                velocity_step_m_per_s=velocity_step_m_per_s,
            )
        )
        if index == 1 or index % 8 == 0:
            print(f"[vector-only helicity] configurations {index:02d}/64", flush=True)

    table_path = statistics_root / "vector_only_helicity_64_configurations.csv"
    pd.DataFrame([_flatten_record(record) for record in records]).to_csv(
        table_path,
        index=False,
    )
    bare_rows = []
    for force_index, force_axis in enumerate("xyz"):
        for variable_index, variable_axis in enumerate("xyz"):
            bare_rows.extend(
                (
                    {
                        "quantity": "bare_780_position_jacobian",
                        "force_component": force_axis,
                        "variable_component": variable_axis,
                        "value": bare_position[force_index, variable_index],
                        "units": "N/m",
                    },
                    {
                        "quantity": "bare_780_velocity_jacobian",
                        "force_component": force_axis,
                        "variable_component": f"v{variable_axis}",
                        "value": bare_velocity[force_index, variable_index],
                        "units": "N s/m",
                    },
                )
            )
    bare_control_path = statistics_root / "bare_780_jacobians.csv"
    pd.DataFrame(bare_rows).to_csv(bare_control_path, index=False)
    print("[vector-only helicity] sampling current/reversed comparisons", flush=True)
    comparison = sample_vector_only_comparison(
        context,
        position_extent_m=comparison_position_extent_m,
        velocity_extent_m_per_s=comparison_velocity_extent_m_per_s,
        sample_count=comparison_sample_count,
    )
    comparison_path = statistics_root / "current_vs_reversed_vector_only_lineouts.csv"
    comparison.to_csv(comparison_path, index=False)
    heatmap_path = _plot_all64(
        records,
        figure_root / "all_64_vector_only_origin_bias.png",
    )
    jacobian_path = _plot_centered_jacobians(
        records,
        bare_velocity,
        figure_root / "centered_vector_only_jacobians.png",
    )
    comparison_figure_path = _plot_current_vs_reversed(
        comparison,
        figure_root / "current_vs_reversed_vector_only_response.png",
    )
    summary_path = statistics_root / "vector_only_helicity_summary.json"
    outputs = [
        table_path,
        bare_control_path,
        comparison_path,
        heatmap_path,
        jacobian_path,
        comparison_figure_path,
        summary_path,
    ]
    counts = _counts(records)
    by_code = {record["combined_code"]: record for record in records}
    restoring_and_damping_codes = [
        record["combined_code"]
        for record in records
        if record["combined_classification"] == "restoring_and_damping"
    ]
    metadata: dict[str, Any] = {
        "schema": "pmot.vector-only-helicity-study.v1",
        "status": "ideal-magic provisional diagnostic; not a production pMOT model",
        "source_full_provisional_model": PROVISIONAL_MODEL_NAME,
        "model": VECTOR_ONLY_MODEL_NAME,
        "effective_detuning_equation": EFFECTIVE_DETUNING_EQUATION,
        "applied_shift_equation": VECTOR_ONLY_SHIFT_EQUATION,
        "force_mechanism": FORCE_MECHANISM,
        "helicity_code_order": list(HELICITY_CODE_ORDER),
        "enumeration_policy": (
            "all 64 sigma-only trapping configurations are recorded; only the "
            "8 incident=retro zero-bias origin equilibria are linearized/classified"
        ),
        "fixed_780_beam_manifest": _beam_manifest(context),
        "fixed_parameters": {
            "cooling_power_w_per_beam": (
                context.apparatus.mot_light.cooling.power_w_per_beam
            ),
            "repump_power_w_per_beam": (
                context.apparatus.mot_light.repump.power_w_per_beam
            ),
            "cooling_detuning_hz": context.apparatus.mot_light.cooling.detuning_hz,
            "trapping_wavelength_m": context.apparatus.trapping_laser.wavelength_m,
            "trapping_power_w_per_path": context.power_w_per_path,
            "trapping_power_source": context.power_source,
            "target_gradient_g_per_cm": context.target_gradient_g_per_cm,
            "external_magnetic_field_t": [0.0, 0.0, 0.0],
            "gravity_included": False,
            "direct_1529_force_included": False,
            "applied_scalar_transition_shift": 0.0,
            "applied_tensor_transition_shift": 0.0,
        },
        "numerical_method": {
            "position_step_m": position_step_m,
            "velocity_step_m_per_s": velocity_step_m_per_s,
            "comparison_position_extent_m": comparison_position_extent_m,
            "comparison_velocity_extent_m_per_s": (
                comparison_velocity_extent_m_per_s
            ),
            "comparison_sample_count": comparison_sample_count,
            "classification": "signs of real parts of complete 3x3 Jacobian eigenvalues",
        },
        "bare_780_control": {
            "position_jacobian_n_per_m": bare_position.tolist(),
            "position_eigenvalues_n_per_m": _complex_list(
                np.linalg.eigvals(bare_position)
            ),
            "velocity_jacobian_n_s_per_m": bare_velocity.tolist(),
            "velocity_eigenvalues_n_s_per_m": _complex_list(
                np.linalg.eigvals(bare_velocity)
            ),
            "velocity_classification": classify_velocity_jacobian(bare_velocity),
            "fallback_quantization_axis": list(BARE_AXIS),
        },
        "counts": counts,
        "key_results": {
            "restoring_and_damping_codes": restoring_and_damping_codes,
            "current": {
                "combined_code": CURRENT_COMBINED_CODE,
                "position": by_code[CURRENT_COMBINED_CODE]["position_classification"],
                "velocity": by_code[CURRENT_COMBINED_CODE]["velocity_classification"],
                "combined": by_code[CURRENT_COMBINED_CODE]["combined_classification"],
            },
            "reversed": {
                "combined_code": REVERSED_COMBINED_CODE,
                "position": by_code[REVERSED_COMBINED_CODE][
                    "position_classification"
                ],
                "velocity": by_code[REVERSED_COMBINED_CODE][
                    "velocity_classification"
                ],
                "combined": by_code[REVERSED_COMBINED_CODE][
                    "combined_classification"
                ],
            },
        },
        "inherited_force_caveat": INHERITED_ABSORPTION_FORCE_CAVEAT,
        "vector_ansatz_caveat": (
            "The differential table does not recover a state-resolved 24-state "
            "Hamiltonian; transition-dependent vector shifts retain the stretched-"
            "reference Zeeman-like ansatz."
        ),
        "zero_axis_caveat": (
            "The centered optical-spin zero uses the +z fallback quantization axis; "
            "nonadiabatic zero-crossing dynamics are absent."
        ),
        "physical_retro_caveat": (
            "The software labels six trapping components independently; physical "
            "retroreflection optics may constrain realizable label pairs."
        ),
        "polarizability_csv": str(context.polarizability_table.source_path),
        "records": records,
        "wall_time_s": perf_counter() - started,
        "outputs": [str(path.resolve()) for path in outputs],
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        "[vector-only helicity] complete: "
        f"counts={counts['centered_combined']}; "
        f"restoring+damping={restoring_and_damping_codes}; summary={summary_path}",
        flush=True,
    )
    return metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the distinct ideal-magic vector-only pMOT helicity audit"
    )
    parser.add_argument("--power-mw-per-path", type=float, default=None)
    parser.add_argument("--target-gradient-g-per-cm", type=float, default=20.0)
    parser.add_argument("--position-step-um", type=float, default=2.0)
    parser.add_argument("--velocity-step-mm-per-s", type=float, default=1.0)
    parser.add_argument("--comparison-position-extent-mm", type=float, default=0.5)
    parser.add_argument("--comparison-velocity-extent-m-per-s", type=float, default=0.5)
    parser.add_argument("--comparison-sample-count", type=int, default=81)
    parser.add_argument("--output-tag", default="vector_only_helicity_20Gpcm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_vector_only_helicity_study(
        power_w_per_path=(
            None if args.power_mw_per_path is None else 1.0e-3 * args.power_mw_per_path
        ),
        target_gradient_g_per_cm=args.target_gradient_g_per_cm,
        position_step_m=1.0e-6 * args.position_step_um,
        velocity_step_m_per_s=1.0e-3 * args.velocity_step_mm_per_s,
        comparison_position_extent_m=1.0e-3 * args.comparison_position_extent_mm,
        comparison_velocity_extent_m_per_s=(
            args.comparison_velocity_extent_m_per_s
        ),
        comparison_sample_count=args.comparison_sample_count,
        output_tag=args.output_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [name for name in globals() if not name.startswith("_")]
