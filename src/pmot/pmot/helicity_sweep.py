"""Exhaustive sigma+/sigma- helicity audit for the provisional no-coil pMOT.

The six configurable trapping-light helicities are ordered as incident x/y/z
followed by retro x/y/z.  This module evaluates all ``2**6`` combinations at
fixed apparatus parameters and records the complete local linearization of the
inherited stationary-atom absorption-force proxy at the origin.

Nothing in this module is a quantitative trapping prediction.  In particular,
the force is the unchanged ground-population-weighted absorption-momentum
proxy of the multilevel rate kernel; conservative Stark force, 1529-nm
scattering/heating, coherent interference, and nonadiabatic dynamics at the
optical-spin zero remain absent.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from dataclasses import replace
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
from ..mot_multilevel.configuration import default_multilevel_mot_config
from ..mot_multilevel.rate_equations import build_rate_equation_model
from .ac_stark import EFFECTIVE_DETUNING_EQUATION
from .ac_stark import PROVISIONAL_MODEL_NAME
from .ac_stark import ProvisionalStarkConfig
from .ac_stark import build_physics_trapping_beams
from .ac_stark import provisional_power_for_target_gradient_w_per_path
from .configuration import build_pmot_cooling_and_repump_beams
from .configuration import default_pmot_apparatus_config
from .configuration import pmot_paths
from .polarizability import load_differential_polarizability_table
from .stark_trajectories import provisional_pmot_observable


HELICITY_CODE_ORDER = (
    "incident_x",
    "incident_y",
    "incident_z",
    "retro_x",
    "retro_y",
    "retro_z",
)
CURRENT_PATH_CODE = "++-"
POSITION_RESTORING_PATH_CODE = "--+"
CURRENT_COMBINED_CODE = CURRENT_PATH_CODE + CURRENT_PATH_CODE
POSITION_RESTORING_COMBINED_CODE = (
    POSITION_RESTORING_PATH_CODE + POSITION_RESTORING_PATH_CODE
)
INHERITED_ABSORPTION_FORCE_CAVEAT = (
    "The force is the authoritative multilevel kernel's ground-population-"
    "weighted absorption-momentum proxy. The same kernel includes stimulated "
    "population links, but their momentum is not subtracted. Its force sign and "
    "magnitude are not yet validated against a consistent two-level limit or "
    "the event engine."
)
OMITTED_PHYSICS = (
    "conservative Stark-gradient force",
    "1529-nm scattering, heating, and loss",
    "coherent standing-wave interference",
    "window and mirror polarization transformations",
    "state-resolved Stark Hamiltonian and nonadiabatic optical-spin-zero dynamics",
)


@dataclass(frozen=True, slots=True)
class HelicitySweepContext:
    """Shared immutable physics inputs for one reproducible sweep."""

    apparatus: Any
    multilevel_config: Any
    model: Any
    polarizability_table: Any
    cooling_repump_beams: tuple[Any, ...]
    power_w_per_path: float
    power_source: str
    target_gradient_g_per_cm: float | None


def path_helicity_codes() -> tuple[str, ...]:
    """Return the eight x/y/z sigma-only path codes in stable order."""

    return tuple("".join(values) for values in itertools.product("+-", repeat=3))


def all_helicity_codes() -> tuple[str, ...]:
    """Return all 64 incident/retro codes in stable row-major order."""

    return tuple(
        incident + retro
        for incident in path_helicity_codes()
        for retro in path_helicity_codes()
    )


def decode_helicity_code(code: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Decode ``ix,iy,iz,rx,ry,rz`` symbols into API helicity labels."""

    if len(code) != 6 or any(symbol not in "+-" for symbol in code):
        raise ValueError("helicity code must contain six '+'/'-' symbols")
    labels = tuple("sigma+" if symbol == "+" else "sigma-" for symbol in code)
    return labels[:3], labels[3:]


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


def build_helicity_sweep_context(
    *,
    power_w_per_path: float | None = None,
    target_gradient_g_per_cm: float = 20.0,
) -> HelicitySweepContext:
    """Build the fixed 27-mW cooling/0.1-mW repump sweep environment."""

    apparatus = default_pmot_apparatus_config()
    multilevel = replace(
        default_multilevel_mot_config(),
        repumper_enabled=True,
        repump_power_w_per_beam=apparatus.mot_light.repump.power_w_per_beam,
    )
    model = build_rate_equation_model(multilevel.natural_linewidth_rad_per_s)
    table = load_differential_polarizability_table()
    if power_w_per_path is None:
        selected_power = provisional_power_for_target_gradient_w_per_path(
            model,
            apparatus.trapping_laser,
            target_gradient_g_per_cm=target_gradient_g_per_cm,
            polarizability_table=table,
        )
        source = "derived provisional stretched-reference vector-gradient proxy"
        recorded_target: float | None = target_gradient_g_per_cm
    else:
        if not np.isfinite(power_w_per_path) or power_w_per_path < 0.0:
            raise ValueError("power_w_per_path must be finite and non-negative")
        selected_power = float(power_w_per_path)
        source = "explicit value"
        recorded_target = None
    cooling_repump = build_pmot_cooling_and_repump_beams(apparatus, multilevel)
    return HelicitySweepContext(
        apparatus=apparatus,
        multilevel_config=multilevel,
        model=model,
        polarizability_table=table,
        cooling_repump_beams=tuple(cooling_repump),
        power_w_per_path=float(selected_power),
        power_source=source,
        target_gradient_g_per_cm=recorded_target,
    )


def _observable_function(
    context: HelicitySweepContext,
    code: str,
) -> tuple[ProvisionalStarkConfig, Callable[[np.ndarray, np.ndarray], Any]]:
    incident, retro = decode_helicity_code(code)
    stark_config = ProvisionalStarkConfig.uniform_power(
        context.power_w_per_path,
        incident_helicities_by_axis=incident,
        retro_helicities_by_axis=retro,
    )
    trapping_beams = build_physics_trapping_beams(
        context.apparatus.trapping_laser,
        stark_config,
    )

    def evaluate(position_m, velocity_m_per_s):
        return provisional_pmot_observable(
            context.model,
            list(context.cooling_repump_beams),
            trapping_beams,
            np.asarray(position_m, dtype=float),
            np.asarray(velocity_m_per_s, dtype=float),
            context.apparatus.trapping_laser,
            stark_config,
            context.multilevel_config,
            polarizability_table=context.polarizability_table,
        )

    return stark_config, evaluate


def _finite_difference_jacobian(
    vector_function: Callable[[np.ndarray], np.ndarray],
    step: float,
) -> np.ndarray:
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("finite-difference step must be finite and positive")
    jacobian = np.empty((3, 3), dtype=float)
    for column in range(3):
        offset = np.zeros(3, dtype=float)
        offset[column] = step
        jacobian[:, column] = (
            np.asarray(vector_function(offset), dtype=float)
            - np.asarray(vector_function(-offset), dtype=float)
        ) / (2.0 * step)
    return jacobian


def classify_position_jacobian(
    jacobian_n_per_m,
    *,
    tolerance_n_per_m: float = 1.0e-23,
) -> str:
    """Classify real parts of local position-Jacobian eigenvalues."""

    real = np.real(np.linalg.eigvals(np.asarray(jacobian_n_per_m, dtype=float)))
    if np.all(real < -tolerance_n_per_m):
        return "restoring"
    if np.all(real > tolerance_n_per_m):
        return "anti_restoring"
    if np.any(real < -tolerance_n_per_m) and np.any(real > tolerance_n_per_m):
        return "saddle"
    return "marginal_or_unresolved"


def classify_velocity_jacobian(
    jacobian_n_s_per_m,
    *,
    tolerance_n_s_per_m: float = 1.0e-27,
) -> str:
    """Classify real parts of local velocity-Jacobian eigenvalues."""

    real = np.real(np.linalg.eigvals(np.asarray(jacobian_n_s_per_m, dtype=float)))
    if np.all(real < -tolerance_n_s_per_m):
        return "damping"
    if np.all(real > tolerance_n_s_per_m):
        return "anti_damping"
    if np.any(real < -tolerance_n_s_per_m) and np.any(real > tolerance_n_s_per_m):
        return "mixed"
    return "marginal_or_unresolved"


def _complex_list(values) -> list[dict[str, float]]:
    return [
        {"real": float(np.real(value)), "imaginary": float(np.imag(value))}
        for value in values
    ]


def evaluate_helicity_configuration(
    context: HelicitySweepContext,
    code: str,
    *,
    position_step_m: float = 2.0e-6,
    velocity_step_m_per_s: float = 1.0e-3,
    field_zero_tolerance_g: float = 1.0e-6,
    force_equilibrium_tolerance_n: float = 1.0e-28,
    position_eigenvalue_tolerance_n_per_m: float = 1.0e-23,
    velocity_eigenvalue_tolerance_n_s_per_m: float = 1.0e-27,
) -> dict[str, Any]:
    """Evaluate origin bias, force, and both 3x3 force Jacobians for one code."""

    stark_config, evaluate = _observable_function(context, code)
    zero = np.zeros(3, dtype=float)
    origin = evaluate(zero, zero)

    def force_at_position(offset):
        return np.asarray(evaluate(offset, zero).rate_equation.force_n, dtype=float)

    def force_at_velocity(offset):
        return np.asarray(evaluate(zero, offset).rate_equation.force_n, dtype=float)

    origin_force = np.asarray(origin.rate_equation.force_n, dtype=float)
    effective_field_t = np.asarray(origin.stark.effective_field_t, dtype=float)
    position_jacobian = _finite_difference_jacobian(
        force_at_position,
        position_step_m,
    )
    velocity_jacobian = _finite_difference_jacobian(
        force_at_velocity,
        velocity_step_m_per_s,
    )
    position_eigenvalues = np.linalg.eigvals(position_jacobian)
    velocity_eigenvalues = np.linalg.eigvals(velocity_jacobian)
    position_signature = classify_position_jacobian(
        position_jacobian,
        tolerance_n_per_m=position_eigenvalue_tolerance_n_per_m,
    )
    velocity_classification = classify_velocity_jacobian(
        velocity_jacobian,
        tolerance_n_s_per_m=velocity_eigenvalue_tolerance_n_s_per_m,
    )
    incident_code = code[:3]
    retro_code = code[3:]
    pair_centered = incident_code == retro_code
    field_zero = float(np.linalg.norm(effective_field_t) * 1.0e4) <= (
        field_zero_tolerance_g
    )
    force_equilibrium = float(np.linalg.norm(origin_force)) <= (
        force_equilibrium_tolerance_n
    )
    centered_equilibrium = pair_centered and field_zero and force_equilibrium
    if not centered_equilibrium:
        origin_position_classification = "not_a_centered_origin_equilibrium"
        dynamic_classification = "not_applicable_origin_not_equilibrium"
    else:
        origin_position_classification = position_signature
        if position_signature == "restoring" and velocity_classification == "damping":
            dynamic_classification = "position_restoring_and_damped"
        elif position_signature == "restoring":
            dynamic_classification = (
                f"position_restoring_but_{velocity_classification}"
            )
        else:
            dynamic_classification = (
                f"position_{position_signature}_and_{velocity_classification}"
            )

    reference = _cycling_transition_index(context.model)
    scalar_hz = float(
        origin.stark.scalar_transition_energy_j[reference] / PLANCK_CONSTANT_J_S
    )
    vector_hz = float(
        origin.stark.vector_transition_energy_j[reference] / PLANCK_CONSTANT_J_S
    )
    tensor_hz = float(
        origin.stark.tensor_transition_energy_j[reference] / PLANCK_CONSTANT_J_S
    )
    total_hz = float(origin.stark.transition_frequency_shift_hz[reference])
    cooling_detuning_hz = (
        context.multilevel_config.cooling_detuning_rad_per_s / (2.0 * np.pi)
    )
    return {
        "combined_code": code,
        "code_order": list(HELICITY_CODE_ORDER),
        "incident_code_xyz": incident_code,
        "retro_code_xyz": retro_code,
        "incident_helicities_xyz": list(stark_config.incident_helicities_by_axis),
        "retro_helicities_xyz": list(stark_config.retro_helicities_by_axis),
        "mismatched_path_count": sum(
            left != right for left, right in zip(incident_code, retro_code)
        ),
        "pair_centered": pair_centered,
        "origin_effective_field_zero": field_zero,
        "origin_force_equilibrium": force_equilibrium,
        "centered_origin_equilibrium": centered_equilibrium,
        "origin_force_n": origin_force.tolist(),
        "origin_force_norm_n": float(np.linalg.norm(origin_force)),
        "origin_effective_field_proxy_t": effective_field_t.tolist(),
        "origin_effective_field_proxy_magnitude_g": float(
            np.linalg.norm(effective_field_t) * 1.0e4
        ),
        "origin_total_trapping_intensity_w_per_m2": (
            origin.stark.total_intensity_w_per_m2
        ),
        "origin_reference_scalar_transition_shift_hz": scalar_hz,
        "origin_reference_vector_transition_shift_hz": vector_hz,
        "origin_reference_tensor_transition_shift_hz": tensor_hz,
        "origin_reference_total_transition_shift_hz": total_hz,
        "origin_reference_effective_cooling_detuning_hz": (
            cooling_detuning_hz - total_hz
        ),
        "position_jacobian_n_per_m": position_jacobian.tolist(),
        "position_jacobian_eigenvalues_n_per_m": _complex_list(position_eigenvalues),
        "position_jacobian_signature": position_signature,
        "origin_position_classification": origin_position_classification,
        "velocity_jacobian_n_s_per_m": velocity_jacobian.tolist(),
        "velocity_jacobian_eigenvalues_n_s_per_m": _complex_list(
            velocity_eigenvalues
        ),
        "velocity_classification": velocity_classification,
        "dynamic_linearization_classification": dynamic_classification,
    }


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested sweep record into a units-explicit CSV row."""

    row = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "code_order",
            "incident_helicities_xyz",
            "retro_helicities_xyz",
            "origin_force_n",
            "origin_effective_field_proxy_t",
            "position_jacobian_n_per_m",
            "position_jacobian_eigenvalues_n_per_m",
            "velocity_jacobian_n_s_per_m",
            "velocity_jacobian_eigenvalues_n_s_per_m",
        }
    }
    for index, label in enumerate("xyz"):
        row[f"origin_force_{label}_n"] = record["origin_force_n"][index]
        row[f"origin_effective_field_proxy_{label}_t"] = record[
            "origin_effective_field_proxy_t"
        ][index]
    for force_index, force_label in enumerate("xyz"):
        for variable_index, variable_label in enumerate("xyz"):
            row[f"dF{force_label}_d{variable_label}_n_per_m"] = record[
                "position_jacobian_n_per_m"
            ][force_index][variable_index]
            row[f"dF{force_label}_dv{variable_label}_n_s_per_m"] = record[
                "velocity_jacobian_n_s_per_m"
            ][force_index][variable_index]
    for index, value in enumerate(record["position_jacobian_eigenvalues_n_per_m"]):
        row[f"position_eigenvalue_{index}_real_n_per_m"] = value["real"]
        row[f"position_eigenvalue_{index}_imaginary_n_per_m"] = value["imaginary"]
    for index, value in enumerate(record["velocity_jacobian_eigenvalues_n_s_per_m"]):
        row[f"velocity_eigenvalue_{index}_real_n_s_per_m"] = value["real"]
        row[f"velocity_eigenvalue_{index}_imaginary_n_s_per_m"] = value[
            "imaginary"
        ]
    return row


def _classification_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    def count(key: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for record in records:
            value = str(record[key])
            values[value] = values.get(value, 0) + 1
        return values

    return {
        "total_configurations": len(records),
        "pair_centered": sum(record["pair_centered"] for record in records),
        "centered_origin_equilibria": sum(
            record["centered_origin_equilibrium"] for record in records
        ),
        "mismatched_path_count": {
            str(count_value): sum(
                record["mismatched_path_count"] == count_value for record in records
            )
            for count_value in range(4)
        },
        "position_jacobian_signature": count("position_jacobian_signature"),
        "origin_position_classification": count("origin_position_classification"),
        "velocity_classification": count("velocity_classification"),
        "dynamic_linearization_classification": count(
            "dynamic_linearization_classification"
        ),
    }


def _plot_all64_heatmaps(records: list[dict[str, Any]], path: Path) -> Path:
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
    for panel in panels:
        panel.set_xticks(range(8), codes, rotation=45, ha="right")
        panel.set_yticks(range(8), codes)
        panel.set_xlabel("Retro path helicities (x y z)")
        panel.set_ylabel("Incident path helicities (x y z)")
        for row, incident in enumerate(codes):
            for column, retro in enumerate(codes):
                record = lookup[(incident, retro)]
                if not record["centered_origin_equilibrium"]:
                    label = "B"
                else:
                    label = {
                        "restoring": "R",
                        "anti_restoring": "A",
                        "saddle": "S",
                        "marginal_or_unresolved": "M",
                    }[record["position_jacobian_signature"]]
                if panel is panels[0]:
                    text_color = "black" if bias[row, column] >= 12.0 else "white"
                else:
                    text_color = "black" if force[row, column] >= 6.8 else "white"
                panel.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                    fontweight="bold",
                )
        for code, color in (
            (CURRENT_PATH_CODE, "#22d3ee"),
            (POSITION_RESTORING_PATH_CODE, "#facc15"),
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
        "Full provisional scalar+vector+tensor proxy at nominal -15 MHz\n"
        "All 64 independent sigma+/sigma- trapping-path configurations\n"
        "B = biased origin; R = restoring; A = anti-restoring; S = saddle; "
        "cyan = current (++-), yellow = reversed (--+)"
    )
    figure.text(
        0.5,
        -0.04,
        "Central stretched-reference detuning is +1.339691 MHz (blue); "
        "not the vector-only design audit; external B and gravity excluded",
        ha="center",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_centered_jacobians(records: list[dict[str, Any]], path: Path) -> Path:
    centered = [record for record in records if record["centered_origin_equilibrium"]]
    centered.sort(key=lambda record: path_helicity_codes().index(record["incident_code_xyz"]))
    labels = [record["incident_code_xyz"] for record in centered]
    position_diagonal = np.asarray(
        [np.diag(record["position_jacobian_n_per_m"]) for record in centered]
    )
    velocity_diagonal = np.asarray(
        [np.diag(record["velocity_jacobian_n_s_per_m"]) for record in centered]
    )
    figure, panels = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)
    colors = ("#b91c1c", "#1d4ed8", "#15803d")
    markers = ("o", "s", "^")
    x = np.arange(len(labels))
    for component, (axis, color, marker) in enumerate(zip("xyz", colors, markers)):
        panels[0].plot(
            x,
            1.0e19 * position_diagonal[:, component],
            marker=marker,
            color=color,
            label=rf"$\partial F_{axis}/\partial {axis}$",
        )
        panels[1].plot(
            x,
            1.0e22 * velocity_diagonal[:, component],
            marker=marker,
            color=color,
            label=rf"$\partial F_{axis}/\partial v_{axis}$",
        )
    panels[0].set(
        title="Position Jacobian diagonal",
        ylabel=r"Slope [$10^{-19}$ N/m]",
    )
    panels[1].set(
        title="Velocity Jacobian diagonal",
        ylabel=r"Slope [$10^{-22}$ N s/m]",
    )
    for panel in panels:
        panel.set_xticks(x, labels)
        panel.set_xlabel("Matched incident = retro helicities (x y z)")
        panel.axhline(0.0, color="0.35", linewidth=0.9)
        panel.grid(alpha=0.25)
        panel.legend(ncol=3, loc="best")
        for code, color in (
            (CURRENT_PATH_CODE, "#0891b2"),
            (POSITION_RESTORING_PATH_CODE, "#ca8a04"),
        ):
            index = labels.index(code)
            panel.axvspan(index - 0.36, index + 0.36, color=color, alpha=0.10)
    figure.suptitle(
        "Full provisional total-shift proxy at nominal -15 MHz: center is blue detuned\n"
        "Eight centered origin equilibria: full Jacobians are stored in CSV/JSON\n"
        "Negative position slope is restoring; negative velocity slope is damping\n"
        "The apparent --+ restoring sign is not the vector-only design recommendation"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def sample_comparison_lineouts(
    context: HelicitySweepContext,
    *,
    position_extent_m: float = 2.0e-3,
    velocity_extent_m_per_s: float = 0.5,
    sample_count: int = 81,
) -> pd.DataFrame:
    """Sample current/reversed spatial Stark and velocity-force comparisons."""

    if sample_count < 3 or sample_count % 2 == 0:
        raise ValueError("sample_count must be an odd integer of at least three")
    reference = _cycling_transition_index(context.model)
    rows: list[dict[str, Any]] = []
    configurations = (
        ("current", CURRENT_COMBINED_CODE),
        ("reversed_position_restoring", POSITION_RESTORING_COMBINED_CODE),
    )
    for role, code in configurations:
        _, evaluate = _observable_function(context, code)
        for component, axis in enumerate("xyz"):
            for coordinate in np.linspace(
                -position_extent_m,
                position_extent_m,
                sample_count,
            ):
                position = np.zeros(3)
                position[component] = coordinate
                observable = evaluate(position, np.zeros(3))
                stark = observable.stark
                rows.append(
                    {
                        "scan_type": "position",
                        "configuration_role": role,
                        "combined_code": code,
                        "axis": axis,
                        "coordinate_m": float(coordinate),
                        "velocity_m_per_s": np.nan,
                        "same_axis_force_proxy_n": float(
                            observable.rate_equation.force_n[component]
                        ),
                        "reference_scalar_transition_shift_hz": float(
                            stark.scalar_transition_energy_j[reference]
                            / PLANCK_CONSTANT_J_S
                        ),
                        "reference_vector_transition_shift_hz": float(
                            stark.vector_transition_energy_j[reference]
                            / PLANCK_CONSTANT_J_S
                        ),
                        "reference_tensor_transition_shift_hz": float(
                            stark.tensor_transition_energy_j[reference]
                            / PLANCK_CONSTANT_J_S
                        ),
                        "reference_total_transition_shift_hz": float(
                            stark.transition_frequency_shift_hz[reference]
                        ),
                        "same_axis_effective_field_proxy_g": float(
                            1.0e4 * stark.effective_field_t[component]
                        ),
                        "effective_field_proxy_magnitude_g": float(
                            1.0e4 * np.linalg.norm(stark.effective_field_t)
                        ),
                    }
                )
            for velocity_component in np.linspace(
                -velocity_extent_m_per_s,
                velocity_extent_m_per_s,
                sample_count,
            ):
                velocity = np.zeros(3)
                velocity[component] = velocity_component
                observable = evaluate(np.zeros(3), velocity)
                rows.append(
                    {
                        "scan_type": "velocity",
                        "configuration_role": role,
                        "combined_code": code,
                        "axis": axis,
                        "coordinate_m": np.nan,
                        "velocity_m_per_s": float(velocity_component),
                        "same_axis_force_proxy_n": float(
                            observable.rate_equation.force_n[component]
                        ),
                        "reference_scalar_transition_shift_hz": np.nan,
                        "reference_vector_transition_shift_hz": np.nan,
                        "reference_tensor_transition_shift_hz": np.nan,
                        "reference_total_transition_shift_hz": np.nan,
                        "same_axis_effective_field_proxy_g": np.nan,
                        "effective_field_proxy_magnitude_g": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _plot_current_vs_reversed_spatial(dataframe: pd.DataFrame, path: Path) -> Path:
    spatial = dataframe[dataframe["scan_type"] == "position"]
    roles = (
        ("current", "Matched (++-): anti-restoring in blue-centered proxy"),
        (
            "reversed_position_restoring",
            "Matched (--+): restoring in blue-centered proxy only",
        ),
    )
    colors = {"x": "#b91c1c", "y": "#1d4ed8", "z": "#15803d"}
    styles = {"x": "-", "y": "--", "z": ":"}
    figure, panels = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    for column, (role, title) in enumerate(roles):
        subset = spatial[spatial["configuration_role"] == role]
        for axis in "xyz":
            axis_data = subset[subset["axis"] == axis]
            coordinate_mm = 1.0e3 * axis_data["coordinate_m"].to_numpy()
            panels[0, column].plot(
                coordinate_mm,
                1.0e21 * axis_data["same_axis_force_proxy_n"],
                color=colors[axis],
                linestyle=styles[axis],
                linewidth=1.9,
                label=axis,
            )
            panels[2, column].plot(
                coordinate_mm,
                axis_data["same_axis_effective_field_proxy_g"],
                color=colors[axis],
                linestyle=styles[axis],
                linewidth=1.9,
                label=axis,
            )
        x_data = subset[subset["axis"] == "x"]
        coordinate_mm = 1.0e3 * x_data["coordinate_m"].to_numpy()
        decomposition = (
            ("scalar", "reference_scalar_transition_shift_hz", "#6b7280", "--"),
            ("vector", "reference_vector_transition_shift_hz", "#7c3aed", "-."),
            ("tensor", "reference_tensor_transition_shift_hz", "#c2410c", ":"),
            ("total", "reference_total_transition_shift_hz", "#111827", "-"),
        )
        for label, field, color, line_style in decomposition:
            panels[1, column].plot(
                coordinate_mm,
                1.0e-6 * x_data[field],
                color=color,
                linestyle=line_style,
                linewidth=2.2 if label == "total" else 1.6,
                label=label,
            )
        panels[0, column].set_title(title)
        panels[0, column].set_ylabel(r"Same-axis force proxy [$10^{-21}$ N]")
        panels[1, column].set_ylabel("Cycling-transition shift [MHz]")
        panels[2, column].set_ylabel(r"Signed vector $B_{\rm eff}$ proxy [G]")
        panels[2, column].set_xlabel("Axis coordinate [mm]")
        for row in range(3):
            panels[row, column].axhline(0.0, color="0.4", linewidth=0.8)
            panels[row, column].axvline(0.0, color="0.4", linewidth=0.8)
            panels[row, column].grid(alpha=0.23)
            panels[row, column].legend(loc="best", ncol=2 if row == 1 else 3)
    figure.suptitle(
        "Full provisional total-shift proxy at nominal -15 MHz: center is blue detuned\n"
        "Helicity reversal: stationary-atom force, transition-shift decomposition, and optical spin\n"
        "780-nm absorption-force proxy; shift is resonance, not intensity/potential; "
        "B_eff is an optical-spin proxy\n"
        "Not the vector-only design audit; no external B or gravity"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def _plot_current_vs_reversed_velocity(dataframe: pd.DataFrame, path: Path) -> Path:
    velocity = dataframe[dataframe["scan_type"] == "velocity"]
    roles = (
        ("current", "Current matched (++-)"),
        ("reversed_position_restoring", "Reversed matched (--+)"),
    )
    colors = {"x": "#b91c1c", "y": "#1d4ed8", "z": "#15803d"}
    styles = {"x": "-", "y": "--", "z": ":"}
    figure, panels = plt.subplots(1, 2, figsize=(14, 5.3), constrained_layout=True)
    for panel, (role, title) in zip(panels, roles):
        subset = velocity[velocity["configuration_role"] == role]
        for axis in "xyz":
            axis_data = subset[subset["axis"] == axis]
            panel.plot(
                axis_data["velocity_m_per_s"],
                1.0e21 * axis_data["same_axis_force_proxy_n"],
                color=colors[axis],
                linestyle=styles[axis],
                linewidth=1.9,
                label=axis,
            )
        panel.axhline(0.0, color="0.4", linewidth=0.8)
        panel.axvline(0.0, color="0.4", linewidth=0.8)
        panel.grid(alpha=0.24)
        panel.legend(ncol=3)
        panel.set(
            title=title,
            xlabel="Same-axis velocity [m/s]",
            ylabel=r"Same-axis 780-nm force proxy [$10^{-21}$ N]",
        )
    figure.suptitle(
        "Full provisional total-shift proxy: both choices are anti-damping because the center is blue detuned\n"
        "Positive origin slope means anti-damping; not the vector-only design audit"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def run_helicity_sweep(
    *,
    power_w_per_path: float | None = None,
    target_gradient_g_per_cm: float = 20.0,
    position_step_m: float = 2.0e-6,
    velocity_step_m_per_s: float = 1.0e-3,
    comparison_position_extent_m: float = 2.0e-3,
    comparison_velocity_extent_m_per_s: float = 0.5,
    comparison_sample_count: int = 81,
    output_tag: str = "helicity_sweep_20Gpcm",
) -> dict[str, Any]:
    """Run, save, and plot the complete fixed-parameter 64-case audit."""

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
        "[pMOT helicity sweep] 64 sigma-only configurations; code order="
        + ",".join(HELICITY_CODE_ORDER),
        flush=True,
    )
    print(
        "[pMOT helicity sweep] fixed parameters: "
        f"cooling={1e3*context.apparatus.mot_light.cooling.power_w_per_beam:.3f} mW/beam, "
        f"repump={1e3*context.apparatus.mot_light.repump.power_w_per_beam:.3f} mW/beam, "
        f"trap={1e3*context.power_w_per_path:.6f} mW/path, external B=0, gravity excluded",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    for index, code in enumerate(all_helicity_codes(), start=1):
        records.append(
            evaluate_helicity_configuration(
                context,
                code,
                position_step_m=position_step_m,
                velocity_step_m_per_s=velocity_step_m_per_s,
            )
        )
        if index == 1 or index % 8 == 0 or index == 64:
            print(
                f"[pMOT helicity sweep] linearizations {index:02d}/64 complete",
                flush=True,
            )

    csv_path = statistics_root / "helicity_sweep_64_configurations.csv"
    pd.DataFrame([_flatten_record(record) for record in records]).to_csv(
        csv_path,
        index=False,
    )
    print("[pMOT helicity sweep] sampling current/reversed comparison lineouts", flush=True)
    comparison = sample_comparison_lineouts(
        context,
        position_extent_m=comparison_position_extent_m,
        velocity_extent_m_per_s=comparison_velocity_extent_m_per_s,
        sample_count=comparison_sample_count,
    )
    comparison_csv_path = statistics_root / "current_vs_reversed_lineouts.csv"
    comparison.to_csv(comparison_csv_path, index=False)

    heatmap_path = _plot_all64_heatmaps(
        records,
        figure_root / "all_64_origin_bias_and_classification.png",
    )
    jacobian_path = _plot_centered_jacobians(
        records,
        figure_root / "centered_position_and_velocity_jacobians.png",
    )
    spatial_comparison_path = _plot_current_vs_reversed_spatial(
        comparison,
        figure_root / "current_vs_reversed_force_shift_and_optical_spin.png",
    )
    velocity_comparison_path = _plot_current_vs_reversed_velocity(
        comparison,
        figure_root / "current_vs_reversed_velocity_response.png",
    )
    summary_path = statistics_root / "helicity_sweep_summary.json"
    output_paths = [
        csv_path,
        comparison_csv_path,
        heatmap_path,
        jacobian_path,
        spatial_comparison_path,
        velocity_comparison_path,
        summary_path,
    ]
    counts = _classification_counts(records)
    record_by_code = {record["combined_code"]: record for record in records}
    centered_restoring_codes = [
        record["combined_code"]
        for record in records
        if record["centered_origin_equilibrium"]
        and record["position_jacobian_signature"] == "restoring"
    ]
    metadata: dict[str, Any] = {
        "schema": "pmot.provisional-helicity-sweep.v1",
        "status": (
            "full provisional scalar+vector+tensor diagnostic at nominal -15 MHz; "
            "the central stretched reference is blue detuned; not the vector-only "
            "design-helicity audit or a validated pMOT trapping prediction"
        ),
        "model": PROVISIONAL_MODEL_NAME,
        "effective_detuning_equation": EFFECTIVE_DETUNING_EQUATION,
        "helicity_code_order": list(HELICITY_CODE_ORDER),
        "helicity_symbol_definition": {
            "+": "sigma+ in the propagation frame; optical-spin sign -1 along k",
            "-": "sigma- in the propagation frame; optical-spin sign +1 along k",
        },
        "independent_degrees_of_freedom": 6,
        "configuration_count": 64,
        "current_combined_code": CURRENT_COMBINED_CODE,
        "position_restoring_combined_code": POSITION_RESTORING_COMBINED_CODE,
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
            "gravity_included_in_force_audit": False,
        },
        "numerical_method": {
            "position_jacobian": "centered finite difference of F at r=0, v=0",
            "position_step_m": position_step_m,
            "velocity_jacobian": "centered finite difference of F at r=0, v=0",
            "velocity_step_m_per_s": velocity_step_m_per_s,
            "position_eigenvalue_tolerance_n_per_m": 1.0e-23,
            "velocity_eigenvalue_tolerance_n_s_per_m": 1.0e-27,
            "origin_effective_field_zero_tolerance_g": 1.0e-6,
            "origin_force_equilibrium_tolerance_n": 1.0e-28,
            "comparison_position_extent_m": comparison_position_extent_m,
            "comparison_velocity_extent_m_per_s": (
                comparison_velocity_extent_m_per_s
            ),
            "comparison_sample_count": comparison_sample_count,
        },
        "classification_semantics": {
            "restoring": "all real parts of position-Jacobian eigenvalues are negative",
            "anti_restoring": "all real parts are positive",
            "saddle": "both negative and positive real parts occur",
            "damping": "all real parts of velocity-Jacobian eigenvalues are negative",
            "anti_damping": "all real parts are positive",
            "not_a_centered_origin_equilibrium": (
                "the origin is biased and its local Jacobian is not an equilibrium-stability test"
            ),
        },
        "classification_counts": counts,
        "key_results": {
            "centered_position_restoring_codes": centered_restoring_codes,
            "current": {
                "combined_code": CURRENT_COMBINED_CODE,
                "position": record_by_code[CURRENT_COMBINED_CODE][
                    "origin_position_classification"
                ],
                "velocity": record_by_code[CURRENT_COMBINED_CODE][
                    "velocity_classification"
                ],
            },
            "reversed": {
                "combined_code": POSITION_RESTORING_COMBINED_CODE,
                "position": record_by_code[POSITION_RESTORING_COMBINED_CODE][
                    "origin_position_classification"
                ],
                "velocity": record_by_code[POSITION_RESTORING_COMBINED_CODE][
                    "velocity_classification"
                ],
            },
            "interpretation": (
                "At these deliberately uncompensated fixed parameters, the common "
                "central Stark shift makes the stretched reference blue detuned. "
                "Reversing all three matched path helicities changes the position "
                "slope sign but does not cure anti-damping. This is not the design "
                "recommendation: the separate vector-only red-detuned audit selects "
                "matched ++- as both position restoring and velocity damping."
            ),
        },
        "provisional_force_caveat": INHERITED_ABSORPTION_FORCE_CAVEAT,
        "effective_field_caveat": (
            "B_eff is only a diagnostic representation of vector optical spin; "
            "it is not passed to the solver as an external magnetic field."
        ),
        "shift_plot_caveat": (
            "The plotted scalar/vector/tensor/total values are stretched-cycling-"
            "transition resonance-frequency shifts, not intensity or potential."
        ),
        "independent_helicity_caveat": (
            "The software sweep treats all six traveling-component labels as "
            "independent. A physical retroreflection train may constrain which "
            "incident/retro propagation-frame helicity pairs are realizable."
        ),
        "quantization_axis_caveat": (
            "At zero optical-spin vector the provisional model uses its configured "
            "fallback quantization axis; nonadiabatic zero-crossing physics is absent."
        ),
        "omitted_physics": list(OMITTED_PHYSICS),
        "polarizability_csv": str(context.polarizability_table.source_path),
        "polarizability_wavelength_range_nm": list(
            context.polarizability_table.wavelength_range_nm
        ),
        "records": records,
        "wall_time_s": perf_counter() - started,
        "outputs": [str(path.resolve()) for path in output_paths],
    }
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        "[pMOT helicity sweep] complete: "
        f"centered={counts['centered_origin_equilibria']}, "
        f"position signatures={counts['position_jacobian_signature']}, "
        f"summary={summary_path}",
        flush=True,
    )
    return metadata


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the exhaustive provisional pMOT trapping-helicity sweep"
    )
    parser.add_argument("--power-mw-per-path", type=float, default=None)
    parser.add_argument("--target-gradient-g-per-cm", type=float, default=20.0)
    parser.add_argument("--position-step-um", type=float, default=2.0)
    parser.add_argument("--velocity-step-mm-per-s", type=float, default=1.0)
    parser.add_argument("--comparison-position-extent-mm", type=float, default=2.0)
    parser.add_argument("--comparison-velocity-extent-m-per-s", type=float, default=0.5)
    parser.add_argument("--comparison-sample-count", type=int, default=81)
    parser.add_argument("--output-tag", default="helicity_sweep_20Gpcm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    run_helicity_sweep(
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
