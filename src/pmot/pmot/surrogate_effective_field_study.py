"""Diagnostics and light-weight MOT tests for the surrogate pMOT field.

The routines here intentionally stop before capture-cross-section or loading-
rate production.  They provide field maps, local force Jacobians, matched
conventional-MOT controls, and a small deterministic trajectory panel.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from ..configuration import GRAVITY_ACCELERATION_M_PER_S2, RB87_MASS_KG
from ..magnetic_fields import default_anti_helmholtz_config
from ..mot_multilevel import (
    RateEquationAtomState,
    RateEquationTrajectoryConfig,
    build_multilevel_mot_beams,
    build_rate_equation_model,
    default_multilevel_mot_config,
    plot_time_diagnostics,
    plot_trajectory_3d,
    rate_equation_observable,
    save_trajectory,
    simulate_rate_equation_trajectory,
)
from .surrogate_effective_field import (
    DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG,
    SurrogateEffectiveFieldConfig,
    inside_surrogate_cell,
    numerical_field_jacobian_t_per_m,
    surrogate_component_intensities_w_per_m2,
    surrogate_effective_field_t,
    surrogate_field_jacobians_t_per_m,
)


OUTPUT_DIRECTORY_NAME = "MOT testing with a strange defined magnetic field"


@dataclass(frozen=True, slots=True)
class DiagnosticTrajectoryCase:
    label: str
    category: str
    position_m: tuple[float, float, float]
    velocity_m_per_s: tuple[float, float, float]
    duration_s: float


def default_diagnostic_trajectory_cases() -> tuple[DiagnosticTrajectoryCase, ...]:
    cases: list[DiagnosticTrajectoryCase] = []
    for coordinate, axis in enumerate("xyz"):
        position = np.zeros(3)
        velocity = np.zeros(3)
        position[coordinate] = 12.0e-3
        velocity[coordinate] = -5.0
        cases.append(
            DiagnosticTrajectoryCase(
                f"incident_{axis}_5mps",
                "incident",
                tuple(position),
                tuple(velocity),
                10.0e-3,
            )
        )
    cases.insert(
        0,
        DiagnosticTrajectoryCase(
            "preloaded_xyz_offset",
            "preloaded",
            (1.0e-3, 1.0e-3, 1.0e-3),
            (0.0, 0.0, 0.0),
            6.0e-3,
        ),
    )
    diagonal_direction = np.ones(3) / np.sqrt(3.0)
    cases.append(
        DiagnosticTrajectoryCase(
            "incident_diagonal_5mps",
            "incident",
            tuple(8.0e-3 * np.ones(3)),
            tuple(-5.0 * diagonal_direction),
            10.0e-3,
        )
    )
    return tuple(cases)


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n", encoding="utf-8")
    return path


def classify_diagnostic_trajectory(
    record,
    *,
    core_radius_m: float = 2.0e-3,
    residence_s: float = 5.0e-3,
) -> dict[str, object]:
    times = np.asarray(record.times_s, dtype=float)
    positions = np.asarray(record.positions_m, dtype=float)
    velocities = np.asarray(record.velocities_m_per_s, dtype=float)
    radii = np.linalg.norm(positions, axis=1)
    inside = radii <= core_radius_m
    entries = int(inside[0]) + int(np.count_nonzero(inside[1:] & ~inside[:-1]))
    maximum_residence = 0.0
    start: float | None = float(times[0]) if inside[0] else None
    for index in range(1, len(times)):
        if inside[index] and not inside[index - 1]:
            start = float(times[index])
        if not inside[index] and inside[index - 1] and start is not None:
            maximum_residence = max(maximum_residence, float(times[index - 1] - start))
            start = None
    if inside[-1] and start is not None:
        maximum_residence = max(maximum_residence, float(times[-1] - start))
    trapped = entries >= 2 or maximum_residence >= residence_s - 1.0e-15
    reason = (
        "two_core_entries"
        if entries >= 2
        else "bounded_core_residence"
        if maximum_residence >= residence_s - 1.0e-15
        else record.termination_reason
        if record.termination_reason != "duration"
        else "duration_not_trapped"
    )
    return {
        "trapped": trapped,
        "classification": reason,
        "core_entry_count": entries,
        "maximum_continuous_core_residence_s": maximum_residence,
        "minimum_radius_m": float(np.min(radii)),
        "final_radius_m": float(radii[-1]),
        "initial_speed_m_per_s": float(np.linalg.norm(velocities[0])),
        "final_speed_m_per_s": float(np.linalg.norm(velocities[-1])),
        "elapsed_s": float(times[-1]),
        "termination_reason": record.termination_reason,
    }


def _field_plane(plane: str, coordinates_m: np.ndarray, config):
    first, second = np.meshgrid(coordinates_m, coordinates_m)
    zeros = np.zeros_like(first)
    if plane == "xy":
        points = np.column_stack((first.ravel(), second.ravel(), zeros.ravel()))
    elif plane == "xz":
        points = np.column_stack((first.ravel(), zeros.ravel(), second.ravel()))
    elif plane == "yz":
        points = np.column_stack((zeros.ravel(), first.ravel(), second.ravel()))
    else:
        raise ValueError("plane must be xy, xz, or yz")
    fields = surrogate_effective_field_t(points, config).reshape((*first.shape, 3))
    return first, second, fields


def _points_in_plane(plane: str, coordinates_m: np.ndarray):
    first, second = np.meshgrid(coordinates_m, coordinates_m)
    zeros = np.zeros_like(first)
    if plane == "xy":
        points = np.column_stack((first.ravel(), second.ravel(), zeros.ravel()))
    elif plane == "xz":
        points = np.column_stack((first.ravel(), zeros.ravel(), second.ravel()))
    elif plane == "yz":
        points = np.column_stack((zeros.ravel(), first.ravel(), second.ravel()))
    else:
        raise ValueError("plane must be xy, xz, or yz")
    return first, second, points


def create_field_diagnostics(
    output_directory: Path,
    config: SurrogateEffectiveFieldConfig | None = None,
    *,
    plane_samples: int = 241,
) -> dict[str, Path]:
    cfg = config or DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG
    figures = output_directory / "figures" / "effective_field"
    data = output_directory / "data" / "effective_field"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    extent = cfg.cell_half_length_m
    coordinates = np.linspace(-extent, extent, plane_samples)
    paths: dict[str, Path] = {}
    plane_maxima_g: dict[str, float] = {}

    line_fields = []
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 11), constrained_layout=True)
    for coordinate_index, axis_name in enumerate("xyz"):
        points = np.zeros((len(coordinates), 3))
        points[:, coordinate_index] = coordinates
        fields = surrogate_effective_field_t(points, cfg)
        line_fields.append(fields)
        axes[coordinate_index, 0].plot(
            1.0e3 * coordinates,
            1.0e4 * fields[:, coordinate_index],
            color=("#b91c1c", "#1d4ed8", "#15803d")[coordinate_index],
        )
        axes[coordinate_index, 1].plot(
            1.0e3 * coordinates,
            1.0e4 * np.linalg.norm(fields, axis=1),
            color="#111827",
        )
        axes[coordinate_index, 0].set(
            title=rf"$B_{axis_name}$ along the {axis_name} axis",
            xlabel=f"{axis_name} [mm]",
            ylabel=f"B{axis_name} [G]",
        )
        axes[coordinate_index, 1].set(
            title=rf"$|\mathbf{{B}}_{{eff}}|$ along the {axis_name} axis",
            xlabel=f"{axis_name} [mm]",
            ylabel="Magnitude [G]",
        )
        for panel in axes[coordinate_index]:
            panel.axvline(0.0, color="0.5", linewidth=0.7)
            panel.axhline(0.0, color="0.5", linewidth=0.7)
            panel.grid(alpha=0.25)
        # asinh preserves the wide dynamic range without introducing the
        # visible derivative kink of a piecewise symlog transform.
        axes[coordinate_index, 0].set_yscale("asinh", linear_width=1.0)
        axes[coordinate_index, 1].set_yscale("log")
    figure.suptitle("Surrogate pMOT effective-field principal-axis cuts")
    paths["axis_linecuts"] = figures / "01_axis_linecuts_and_magnitudes.png"
    figure.savefig(paths["axis_linecuts"], dpi=200)
    plt.close(figure)

    line_fields_array = np.asarray(line_fields)
    np.savez_compressed(
        data / "axis_linecuts.npz",
        coordinate_m=coordinates,
        field_t=line_fields_array,
    )
    paths["axis_linecuts_data"] = data / "axis_linecuts.npz"

    central_coordinates = np.linspace(-2.0e-3, 2.0e-3, 241)
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 11), constrained_layout=True)
    for coordinate_index, axis_name in enumerate("xyz"):
        points = np.zeros((len(central_coordinates), 3))
        points[:, coordinate_index] = central_coordinates
        fields = surrogate_effective_field_t(points, cfg)
        axes[coordinate_index, 0].plot(
            1.0e3 * central_coordinates,
            1.0e4 * fields[:, coordinate_index],
            color=("#b91c1c", "#1d4ed8", "#15803d")[coordinate_index],
        )
        axes[coordinate_index, 1].plot(
            1.0e3 * central_coordinates,
            1.0e4 * np.linalg.norm(fields, axis=1),
            color="#111827",
        )
        axes[coordinate_index, 0].set(
            title=rf"Central $B_{axis_name}$ along {axis_name}",
            xlabel=f"{axis_name} [mm]",
            ylabel=f"B{axis_name} [G]",
        )
        axes[coordinate_index, 1].set(
            title=rf"Central $|\mathbf{{B}}_{{eff}}|$ along {axis_name}",
            xlabel=f"{axis_name} [mm]",
            ylabel="Magnitude [G]",
        )
        for panel in axes[coordinate_index]:
            panel.axvline(0.0, color="0.5", linewidth=0.7)
            panel.axhline(0.0, color="0.5", linewidth=0.7)
            panel.grid(alpha=0.25)
    figure.suptitle("Central ±2 mm surrogate pMOT effective-field cuts")
    paths["central_axis_linecuts"] = figures / "01b_central_axis_linecuts.png"
    figure.savefig(paths["central_axis_linecuts"], dpi=200)
    plt.close(figure)

    gradient_coordinates = np.linspace(-3.0e-3, 3.0e-3, 181)
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), constrained_layout=True)
    gradient_colors = ("#b91c1c", "#1d4ed8", "#15803d")
    gradient_targets = (10.0, 10.0, -20.0)
    for coordinate_index, axis_name in enumerate("xyz"):
        points = np.zeros((len(gradient_coordinates), 3))
        points[:, coordinate_index] = gradient_coordinates
        jacobians = surrogate_field_jacobians_t_per_m(
            points,
            step_m=1.0e-6,
            config=cfg,
        )
        diagonal_gradient = 100.0 * jacobians[:, coordinate_index, coordinate_index]
        axes[coordinate_index].plot(
            1.0e3 * gradient_coordinates,
            diagonal_gradient,
            color=gradient_colors[coordinate_index],
        )
        axes[coordinate_index].axhline(
            gradient_targets[coordinate_index],
            color="0.35",
            linestyle="--",
            linewidth=0.9,
            label="ideal local target",
        )
        axes[coordinate_index].axvline(0.0, color="0.5", linewidth=0.7)
        axes[coordinate_index].set(
            title=rf"$\partial B_{axis_name}/\partial {axis_name}$ along {axis_name}",
            xlabel=f"{axis_name} [mm]",
            ylabel="gradient [G/cm]",
        )
        axes[coordinate_index].grid(alpha=0.25)
        axes[coordinate_index].legend(frameon=False)
    figure.suptitle("Centered surrogate-field principal gradients (central ±3 mm)")
    paths["central_axis_gradients"] = figures / "01c_central_axis_gradients.png"
    figure.savefig(paths["central_axis_gradients"], dpi=200)
    plt.close(figure)

    for plane in ("xy", "xz", "yz"):
        first, second, fields = _field_plane(plane, coordinates, cfg)
        magnitude = np.linalg.norm(fields, axis=2)
        plane_maxima_g[plane] = float(1.0e4 * np.max(magnitude))
        arrays_g = tuple(1.0e4 * fields[:, :, index] for index in range(3)) + (
            1.0e4 * magnitude,
        )
        figure, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
        labels = (r"$B_x$ [G]", r"$B_y$ [G]", r"$B_z$ [G]", r"$|\mathbf{B}|$ [G]")
        for index, (panel, values, label) in enumerate(
            zip(axes.flat[:4], arrays_g, labels)
        ):
            if index < 3:
                maximum = max(1.0e-15, float(np.max(np.abs(values))))
                normalizer = colors.SymLogNorm(
                    linthresh=1.0,
                    linscale=0.8,
                    vmin=-maximum,
                    vmax=maximum,
                    base=10.0,
                )
                cmap = "coolwarm"
            else:
                positive = values[values > 0.0]
                normalizer = colors.LogNorm(
                    vmin=max(1.0e-6, float(np.min(positive))),
                    vmax=max(1.0e-5, float(np.max(values))),
                )
                cmap = "magma"
            image = panel.pcolormesh(
                1.0e3 * first,
                1.0e3 * second,
                values,
                shading="auto",
                cmap=cmap,
                norm=normalizer,
            )
            figure.colorbar(image, ax=panel, label=label)
            panel.set_title(f"{label} in {plane.upper()}")
            panel.set_aspect("equal")
        component_indices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[plane]
        stride = max(1, plane_samples // 25)
        q_panel = axes.flat[4]
        q_panel.quiver(
            1.0e3 * first[::stride, ::stride],
            1.0e3 * second[::stride, ::stride],
            fields[::stride, ::stride, component_indices[0]],
            fields[::stride, ::stride, component_indices[1]],
            1.0e4 * magnitude[::stride, ::stride],
            cmap="viridis",
            norm=colors.LogNorm(
                vmin=max(1.0e-6, float(np.min(1.0e4 * magnitude[magnitude > 0.0]))),
                vmax=max(1.0e-5, float(np.max(1.0e4 * magnitude))),
            ),
            pivot="mid",
        )
        q_panel.set_title("In-plane direction; color is |B|")
        q_panel.set_aspect("equal")
        unit = np.divide(
            fields,
            magnitude[:, :, np.newaxis],
            out=np.zeros_like(fields),
            where=magnitude[:, :, np.newaxis] > 1.0e-15,
        )
        u_panel = axes.flat[5]
        u_panel.streamplot(
            1.0e3 * coordinates,
            1.0e3 * coordinates,
            unit[:, :, component_indices[0]],
            unit[:, :, component_indices[1]],
            color=1.0e4 * magnitude,
            cmap="viridis",
            density=1.15,
        )
        u_panel.set_title("Quantization-axis direction streamlines")
        u_panel.set_aspect("equal")
        for panel in axes.flat:
            panel.set_xlabel(f"{plane[0]} [mm]")
            panel.set_ylabel(f"{plane[1]} [mm]")
        figure.suptitle(f"Surrogate pMOT effective field in the {plane.upper()} plane")
        paths[f"plane_{plane}"] = figures / f"02_field_slice_{plane}.png"
        figure.savefig(paths[f"plane_{plane}"], dpi=200)
        plt.close(figure)
        np.savez_compressed(
            data / f"field_slice_{plane}.npz",
            coordinate_1_m=coordinates,
            coordinate_2_m=coordinates,
            field_t=fields,
            magnitude_t=magnitude,
        )
        paths[f"plane_{plane}_data"] = data / f"field_slice_{plane}.npz"

    gradient_plane_summary: dict[str, object] = {}
    gradient_labels = (
        r"$\partial B_x/\partial x$ [G/cm]",
        r"$\partial B_y/\partial y$ [G/cm]",
        r"$\partial B_z/\partial z$ [G/cm]",
        r"$\nabla\!\cdot\!\mathbf{B}_{eff}$ [G/cm]",
        "off-diagonal Jacobian norm [G/cm]",
        "Jacobian Frobenius norm [G/cm]",
    )
    for plane in ("xy", "xz", "yz"):
        first, second, points = _points_in_plane(plane, gradient_coordinates)
        jacobians = surrogate_field_jacobians_t_per_m(
            points,
            step_m=1.0e-6,
            config=cfg,
        ).reshape((*first.shape, 3, 3))
        jacobians_g_per_cm = 100.0 * jacobians
        diagonal = np.diagonal(jacobians_g_per_cm, axis1=2, axis2=3)
        divergence = np.trace(jacobians_g_per_cm, axis1=2, axis2=3)
        off_diagonal = jacobians_g_per_cm.copy()
        for index in range(3):
            off_diagonal[:, :, index, index] = 0.0
        off_diagonal_norm = np.linalg.norm(off_diagonal, axis=(2, 3))
        jacobian_norm = np.linalg.norm(jacobians_g_per_cm, axis=(2, 3))
        arrays = (
            diagonal[:, :, 0],
            diagonal[:, :, 1],
            diagonal[:, :, 2],
            divergence,
            off_diagonal_norm,
            jacobian_norm,
        )
        figure, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
        for panel_index, (panel, values, label) in enumerate(
            zip(axes.flat, arrays, gradient_labels)
        ):
            if panel_index < 4:
                limit = max(1.0e-12, float(np.max(np.abs(values))))
                normalizer = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
                cmap = "coolwarm"
            else:
                normalizer = colors.Normalize(
                    vmin=0.0,
                    vmax=max(1.0e-12, float(np.max(values))),
                )
                cmap = "viridis"
            image = panel.pcolormesh(
                1.0e3 * first,
                1.0e3 * second,
                values,
                shading="auto",
                cmap=cmap,
                norm=normalizer,
            )
            figure.colorbar(image, ax=panel, label=label)
            panel.set(
                xlabel=f"{plane[0]} [mm]",
                ylabel=f"{plane[1]} [mm]",
                title=label,
            )
            panel.set_aspect("equal")
        figure.suptitle(
            f"Centered surrogate magnetic-field gradients in the {plane.upper()} plane "
            "(central ±3 mm)"
        )
        paths[f"gradient_plane_{plane}"] = figures / f"04_gradient_slice_{plane}.png"
        figure.savefig(paths[f"gradient_plane_{plane}"], dpi=200)
        plt.close(figure)
        np.savez_compressed(
            data / f"gradient_slice_{plane}.npz",
            coordinate_1_m=gradient_coordinates,
            coordinate_2_m=gradient_coordinates,
            jacobian_g_per_cm=jacobians_g_per_cm,
            divergence_g_per_cm=divergence,
        )
        paths[f"gradient_plane_{plane}_data"] = data / f"gradient_slice_{plane}.npz"
        gradient_plane_summary[plane] = {
            "diagonal_min_g_per_cm": np.min(diagonal, axis=(0, 1)),
            "diagonal_max_g_per_cm": np.max(diagonal, axis=(0, 1)),
            "maximum_abs_divergence_g_per_cm": float(np.max(np.abs(divergence))),
            "maximum_off_diagonal_norm_g_per_cm": float(np.max(off_diagonal_norm)),
        }

    grid = np.unique(
        np.concatenate(
            (
                np.linspace(-extent, extent, 13),
                (cfg.forward_waist_position_m, cfg.return_waist_position_m, 0.0),
            )
        )
    )
    x, y, z = np.meshgrid(grid, grid, grid, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    fields = surrogate_effective_field_t(points, cfg)
    magnitude = np.linalg.norm(fields, axis=1)
    unit = np.divide(
        fields,
        magnitude[:, np.newaxis],
        out=np.zeros_like(fields),
        where=magnitude[:, np.newaxis] > 1.0e-15,
    )
    figure = plt.figure(figsize=(10, 8.5), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    maximum_g = max(1.0e-15, float(np.max(1.0e4 * magnitude)))
    positive_g = 1.0e4 * magnitude[magnitude > 0.0]
    magnitude_normalizer = colors.LogNorm(
        vmin=max(1.0e-6, float(np.min(positive_g))),
        vmax=maximum_g,
    )
    rgba = plt.get_cmap("viridis")(magnitude_normalizer(1.0e4 * magnitude))
    axis.quiver(
        *(1.0e3 * points).T,
        *unit.T,
        length=1.35,
        normalize=False,
        colors=rgba,
        linewidth=0.65,
        arrow_length_ratio=0.28,
    )
    mappable = plt.cm.ScalarMappable(norm=magnitude_normalizer, cmap="viridis")
    figure.colorbar(mappable, ax=axis, shrink=0.68, pad=0.08, label="|B| [G]")
    extent_mm = 1.0e3 * extent
    axis.set(
        xlim=(-extent_mm, extent_mm),
        ylim=(-extent_mm, extent_mm),
        zlim=(-extent_mm, extent_mm),
        xlabel="x [mm]",
        ylabel="y [mm]",
        zlabel="z [mm]",
        title="3D direction and magnitude of the surrogate pMOT effective field",
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))
    paths["field_3d"] = figures / "03_field_vector_cloud_3d.png"
    figure.savefig(paths["field_3d"], dpi=210)
    plt.close(figure)

    jacobian = numerical_field_jacobian_t_per_m(config=cfg)
    origin = surrogate_effective_field_t(np.zeros(3), cfg)
    summary = {
        "schema": "pmot.surrogate-effective-field-diagnostics.v1",
        "interpretation": (
            "stretched-transition-equivalent surrogate field; not a state-resolved "
            "AC-Stark Hamiltonian"
        ),
        "configuration": cfg.metadata(),
        "origin_field_t": origin,
        "origin_field_g": 1.0e4 * origin,
        "origin_jacobian_t_per_m": jacobian,
        "origin_jacobian_g_per_cm": 100.0 * jacobian,
        "origin_divergence_t_per_m": float(np.trace(jacobian)),
        "origin_divergence_g_per_cm": float(100.0 * np.trace(jacobian)),
        "gradient_plane_extent_m": 3.0e-3,
        "gradient_plane_summary": gradient_plane_summary,
        "plane_maximum_sampled_field_g": plane_maxima_g,
        "maximum_fine_plane_sampled_field_g": float(max(plane_maxima_g.values())),
        "maximum_waist_augmented_3d_grid_field_t": float(np.max(magnitude)),
        "maximum_waist_augmented_3d_grid_field_g": float(1.0e4 * np.max(magnitude)),
    }
    paths["summary"] = _write_json(data / "field_diagnostic_summary.json", summary)
    return paths


def _optical_force(
    position_m,
    velocity_m_per_s,
    *,
    model,
    beams,
    config,
    coil_config,
    magnetic_field_function=None,
) -> np.ndarray:
    observable = rate_equation_observable(
        model,
        beams,
        tuple(position_m),
        tuple(velocity_m_per_s),
        coil_config,
        config,
        magnetic_field_function=magnetic_field_function,
    )
    return np.asarray(observable.force_n, dtype=float)


def create_force_diagnostics(output_directory: Path) -> dict[str, object]:
    figures = output_directory / "figures" / "force"
    data = output_directory / "data" / "force"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    config = replace(default_multilevel_mot_config(), include_gravity=False)
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=10.0)
    positions = np.linspace(-2.0e-3, 2.0e-3, 33)
    velocities = np.linspace(-2.0, 2.0, 33)
    models = {
        "surrogate": (None, surrogate_effective_field_t),
        "coil": (coil, None),
    }
    force_data: dict[str, dict[str, list]] = {}
    figure, axes = plt.subplots(3, 2, figsize=(12.5, 11), constrained_layout=True)
    for model_name, (coil_config, field_function) in models.items():
        force_data[model_name] = {"position_force_n": [], "velocity_force_n": []}
        for coordinate, axis_name in enumerate("xyz"):
            position_forces = []
            velocity_forces = []
            for value in positions:
                position = np.zeros(3)
                position[coordinate] = value
                position_forces.append(
                    _optical_force(
                        position,
                        np.zeros(3),
                        model=model,
                        beams=beams,
                        config=config,
                        coil_config=coil_config,
                        magnetic_field_function=field_function,
                    )[coordinate]
                )
            for value in velocities:
                velocity = np.zeros(3)
                velocity[coordinate] = value
                velocity_forces.append(
                    _optical_force(
                        np.zeros(3),
                        velocity,
                        model=model,
                        beams=beams,
                        config=config,
                        coil_config=coil_config,
                        magnetic_field_function=field_function,
                    )[coordinate]
                )
            force_data[model_name]["position_force_n"].append(position_forces)
            force_data[model_name]["velocity_force_n"].append(velocity_forces)
            style = "-" if model_name == "surrogate" else "--"
            axes[coordinate, 0].plot(
                1.0e3 * positions,
                1.0e21 * np.asarray(position_forces),
                style,
                label=model_name,
            )
            axes[coordinate, 1].plot(
                velocities,
                1.0e21 * np.asarray(velocity_forces),
                style,
                label=model_name,
            )
            axes[coordinate, 0].set(
                title=rf"Restoring check: $F_{axis_name}({axis_name})$",
                xlabel=f"{axis_name} [mm]",
                ylabel=f"F{axis_name} [zN]",
            )
            axes[coordinate, 1].set(
                title=rf"Damping check: $F_{axis_name}(v_{axis_name})$",
                xlabel=f"v{axis_name} [m/s]",
                ylabel=f"F{axis_name} [zN]",
            )
    for panel in axes.flat:
        panel.axhline(0.0, color="0.5", linewidth=0.7)
        panel.axvline(0.0, color="0.5", linewidth=0.7)
        panel.grid(alpha=0.25)
        panel.legend()
    figure.suptitle("24-state Section-12 radiation force: surrogate field vs 10 G/cm coil")
    force_plot = figures / "04_restoring_and_damping_comparison.png"
    figure.savefig(force_plot, dpi=200)
    plt.close(figure)

    def force_jacobian(
        model_name: str,
        variable: str,
        step: float,
        *,
        position_center=None,
        velocity_center=None,
    ) -> np.ndarray:
        coil_config, field_function = models[model_name]
        position_center = np.zeros(3) if position_center is None else np.asarray(position_center)
        velocity_center = np.zeros(3) if velocity_center is None else np.asarray(velocity_center)
        result = np.empty((3, 3))
        for coordinate in range(3):
            increment = np.zeros(3)
            increment[coordinate] = step
            if variable == "position":
                f_plus = _optical_force(
                    position_center + increment, velocity_center,
                    model=model, beams=beams, config=config,
                    coil_config=coil_config, magnetic_field_function=field_function,
                )
                f_minus = _optical_force(
                    position_center - increment, velocity_center,
                    model=model, beams=beams, config=config,
                    coil_config=coil_config, magnetic_field_function=field_function,
                )
            else:
                f_plus = _optical_force(
                    position_center, velocity_center + increment,
                    model=model, beams=beams, config=config,
                    coil_config=coil_config, magnetic_field_function=field_function,
                )
                f_minus = _optical_force(
                    position_center, velocity_center - increment,
                    model=model, beams=beams, config=config,
                    coil_config=coil_config, magnetic_field_function=field_function,
                )
            result[:, coordinate] = (f_plus - f_minus) / (2.0 * step)
        return result

    def find_equilibrium(model_name: str, *, include_gravity: bool) -> dict[str, object]:
        coil_config, field_function = models[model_name]
        position = np.zeros(3)
        if model_name == "surrogate":
            # The measured field offset predicts a common positive-axis shift.
            position[:] = 0.6e-3
        gravity_force = (
            RB87_MASS_KG * np.asarray(GRAVITY_ACCELERATION_M_PER_S2)
            if include_gravity
            else np.zeros(3)
        )
        converged = False
        iterations = 0
        for iterations in range(1, 21):
            force = _optical_force(
                position, np.zeros(3), model=model, beams=beams, config=config,
                coil_config=coil_config, magnetic_field_function=field_function,
            ) + gravity_force
            if np.linalg.norm(force) <= 1.0e-27:
                converged = True
                break
            jacobian = force_jacobian(
                model_name,
                "position",
                1.0e-6,
                position_center=position,
            )
            try:
                step = np.linalg.solve(jacobian, -force)
            except np.linalg.LinAlgError:
                break
            step_norm = float(np.linalg.norm(step))
            if step_norm > 0.5e-3:
                step *= 0.5e-3 / step_norm
            position += step
            if not inside_surrogate_cell(position):
                break
        optical_force = _optical_force(
            position, np.zeros(3), model=model, beams=beams, config=config,
            coil_config=coil_config, magnetic_field_function=field_function,
        )
        residual = optical_force + gravity_force
        position_jacobian = force_jacobian(
            model_name, "position", 1.0e-6, position_center=position
        )
        velocity_jacobian = force_jacobian(
            model_name, "velocity", 1.0e-3, position_center=position
        )
        position_eigenvalues = np.linalg.eigvals(position_jacobian)
        velocity_eigenvalues = np.linalg.eigvals(velocity_jacobian)
        return {
            "include_gravity": include_gravity,
            "converged": converged,
            "iteration_count": iterations,
            "position_m": position,
            "position_mm": 1.0e3 * position,
            "optical_force_n": optical_force,
            "gravity_force_n": gravity_force,
            "net_force_residual_n": residual,
            "net_force_residual_norm_n": float(np.linalg.norm(residual)),
            "magnetic_field_t": (
                surrogate_effective_field_t(position)
                if model_name == "surrogate"
                else np.asarray(rate_equation_observable(
                    model, beams, tuple(position), (0.0, 0.0, 0.0), coil,
                    config,
                ).magnetic_field_t)
            ),
            "position_force_jacobian_n_per_m": position_jacobian,
            "velocity_force_jacobian_n_per_m_per_s": velocity_jacobian,
            "position_jacobian_eigenvalues_n_per_m": position_eigenvalues.real,
            "velocity_jacobian_eigenvalues_n_per_m_per_s": velocity_eigenvalues.real,
            "all_position_modes_restoring": bool(np.all(position_eigenvalues.real < 0.0)),
            "all_velocity_modes_damping": bool(np.all(velocity_eigenvalues.real < 0.0)),
        }

    summary: dict[str, object] = {
        "schema": "pmot.surrogate-field-force-diagnostics.v1",
        "solver": "physical Section-12 24-state deterministic mean force",
        "gravity_included": False,
        "cooling_power_w_per_beam": config.cooling_power_w_per_beam,
        "repump_power_w_per_beam": config.repump_power_w_per_beam,
        "cooling_detuning_rad_per_s": config.cooling_detuning_rad_per_s,
        "models": {},
    }
    for model_name in models:
        position_jacobian = force_jacobian(model_name, "position", 1.0e-5)
        velocity_jacobian = force_jacobian(model_name, "velocity", 1.0e-3)
        position_eigenvalues = np.linalg.eigvals(position_jacobian)
        velocity_eigenvalues = np.linalg.eigvals(velocity_jacobian)
        summary["models"][model_name] = {
            "origin_force_n": _optical_force(
                np.zeros(3), np.zeros(3), model=model, beams=beams, config=config,
                coil_config=models[model_name][0],
                magnetic_field_function=models[model_name][1],
            ),
            "position_force_jacobian_n_per_m": position_jacobian,
            "velocity_force_jacobian_n_per_m_per_s": velocity_jacobian,
            "position_jacobian_eigenvalues_n_per_m": position_eigenvalues.real,
            "velocity_jacobian_eigenvalues_n_per_m_per_s": velocity_eigenvalues.real,
            "all_position_modes_restoring": bool(np.all(position_eigenvalues.real < 0.0)),
            "all_velocity_modes_damping": bool(np.all(velocity_eigenvalues.real < 0.0)),
            "optical_equilibrium": find_equilibrium(model_name, include_gravity=False),
            "gravity_balanced_equilibrium": find_equilibrium(model_name, include_gravity=True),
        }
    np.savez_compressed(
        data / "force_curves.npz",
        positions_m=positions,
        velocities_m_per_s=velocities,
        surrogate_position_force_n=np.asarray(force_data["surrogate"]["position_force_n"]),
        surrogate_velocity_force_n=np.asarray(force_data["surrogate"]["velocity_force_n"]),
        coil_position_force_n=np.asarray(force_data["coil"]["position_force_n"]),
        coil_velocity_force_n=np.asarray(force_data["coil"]["velocity_force_n"]),
    )
    summary_path = _write_json(data / "force_diagnostic_summary.json", summary)
    return {"plot": force_plot, "data": data / "force_curves.npz", "summary": summary_path, **summary}


def _plot_paired_trajectory(case, records, output_path: Path) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for model_name, record in records.items():
        time_ms = 1.0e3 * np.asarray(record.times_s)
        positions = np.asarray(record.positions_m)
        velocities = np.asarray(record.velocities_m_per_s)
        style = "-" if model_name == "surrogate" else "--"
        axes[0, 0].plot(time_ms, 1.0e3 * np.linalg.norm(positions, axis=1), style, label=model_name)
        axes[0, 1].plot(time_ms, np.linalg.norm(velocities, axis=1), style, label=model_name)
        for coordinate, color in enumerate(("#b91c1c", "#1d4ed8", "#15803d")):
            axes[1, 0].plot(time_ms, 1.0e3 * positions[:, coordinate], style, color=color, alpha=0.8)
            axes[1, 1].plot(time_ms, velocities[:, coordinate], style, color=color, alpha=0.8)
    axes[0, 0].set(title="Radius", ylabel="Radius [mm]")
    axes[0, 1].set(title="Speed", ylabel="Speed [m/s]")
    axes[1, 0].set(title="Cartesian position", ylabel="Position [mm]")
    axes[1, 1].set(title="Cartesian velocity", ylabel="Velocity [m/s]")
    for panel in axes.flat:
        panel.set_xlabel("Time [ms]")
        panel.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[0, 1].legend()
    figure.suptitle(f"Matched field comparison: {case.label}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    plt.close(figure)
    return output_path


def _simulate_trajectory_task(payload):
    case, model_name, time_step_s, model, beams, config, coil = payload
    coil_config = None if model_name == "surrogate" else coil
    field_function = surrogate_effective_field_t if model_name == "surrogate" else None
    record = simulate_rate_equation_trajectory(
        RateEquationAtomState(case.position_m, case.velocity_m_per_s),
        case.duration_s,
        coil_config,
        beams=beams,
        model=model,
        config=config,
        trajectory_config=RateEquationTrajectoryConfig(
            time_step_s=time_step_s,
            escape_radius_m=30.0e-3,
        ),
        magnetic_field_function=field_function,
        spatial_domain_function=inside_surrogate_cell,
    )
    return case, model_name, record


def run_diagnostic_trajectory_panel(
    output_directory: Path,
    *,
    time_step_s: float = 5.0e-6,
    cases: tuple[DiagnosticTrajectoryCase, ...] | None = None,
    worker_count: int = 3,
) -> dict[str, object]:
    selected_cases = cases or default_diagnostic_trajectory_cases()
    figures = output_directory / "figures" / "trajectories"
    data = output_directory / "data" / "trajectories"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    config = default_multilevel_mot_config()
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    coil = default_anti_helmholtz_config(target_gradient_g_per_cm=10.0)
    model_names = ("surrogate", "coil")
    summaries: list[dict[str, object]] = []
    records_by_case = {case.label: {} for case in selected_cases}
    payloads = [
        (case, model_name, time_step_s, model, beams, config, coil)
        for case in selected_cases
        for model_name in model_names
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for completed, (case, model_name, record) in enumerate(
            executor.map(_simulate_trajectory_task, payloads),
            start=1,
        ):
            print(
                f"[surrogate field study] trajectory {completed}/{len(payloads)} "
                f"{case.label} model={model_name} complete",
                flush=True,
            )
            records_by_case[case.label][model_name] = record
            classification = classify_diagnostic_trajectory(record)
            summaries.append(
                {
                    "case": case.label,
                    "category": case.category,
                    "model": model_name,
                    "initial_position_m": case.position_m,
                    "initial_velocity_m_per_s": case.velocity_m_per_s,
                    "duration_s": case.duration_s,
                    "time_step_s": time_step_s,
                    **classification,
                }
            )
            save_trajectory(
                record,
                model,
                beams,
                data / f"{case.label}_{model_name}",
                metadata={
                    "field_model": model_name,
                    "surrogate_field_specification": "PMOT_EFFECTIVE_FIELD_CODEX.md",
                    "cell_wall_policy": "terminate at first stored point outside the closed cubic cell",
                    "trajectory_case": asdict(case),
                },
            )
    for case in selected_cases:
        records = records_by_case[case.label]
        _plot_paired_trajectory(
            case,
            records,
            figures / f"trajectory_comparison_{case.label}.png",
        )

    csv_path = data / "trajectory_panel_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    aggregate = {}
    for model_name in model_names:
        rows = [row for row in summaries if row["model"] == model_name]
        trapped = sum(bool(row["trapped"]) for row in rows)
        aggregate[model_name] = {
            "case_count": len(rows),
            "trapped_count": trapped,
            "diagnostic_trapped_fraction": trapped / len(rows),
            "wall_loss_count": sum(row["termination_reason"] == "wall_loss" for row in rows),
            "incident_case_count": sum(row["category"] == "incident" for row in rows),
            "incident_trapped_count": sum(
                bool(row["trapped"]) and row["category"] == "incident" for row in rows
            ),
            "preloaded_case_count": sum(row["category"] == "preloaded" for row in rows),
            "preloaded_trapped_count": sum(
                bool(row["trapped"]) and row["category"] == "preloaded" for row in rows
            ),
        }
    summary_payload = {
        "schema": "pmot.surrogate-field-trajectory-panel.v1",
        "interpretation": (
            "small deterministic diagnostic panel, not a capture cross section or loading rate"
        ),
        "solver": "physical Section-12 24-state deterministic mean-force MOT",
        "gravity_included": config.include_gravity,
        "time_step_s": time_step_s,
        "cell_half_length_m": DEFAULT_SURROGATE_EFFECTIVE_FIELD_CONFIG.cell_half_length_m,
        "capture_criterion": "two core entries or continuous 2-mm-core residence for 5 ms",
        "aggregate": aggregate,
        "cases": summaries,
    }
    summary_path = _write_json(data / "trajectory_panel_summary.json", summary_payload)

    representative = selected_cases[min(1, len(selected_cases) - 1)].label
    for model_name, record in records_by_case[representative].items():
        figure = plot_trajectory_3d(
            record,
            beams,
            figures / f"representative_3d_{representative}_{model_name}.png",
            title=f"{model_name}: {representative}",
            spatial_extent_mm=15.0,
        )
        plt.close(figure)
        figure = plot_time_diagnostics(
            record,
            model,
            beams,
            figures / f"representative_time_{representative}_{model_name}.png",
        )
        plt.close(figure)
    return {
        "csv": csv_path,
        "summary": summary_path,
        "aggregate": aggregate,
        "cases": summaries,
        "records": records_by_case,
    }


def run_timestep_audit(
    output_directory: Path,
    *,
    case: DiagnosticTrajectoryCase | None = None,
    time_steps_s=(5.0e-6, 2.5e-6),
) -> dict[str, object]:
    selected = case or DiagnosticTrajectoryCase(
        "incident_x_5mps_timestep_audit",
        "incident",
        (12.0e-3, 0.0, 0.0),
        (-5.0, 0.0, 0.0),
        10.0e-3,
    )
    config = default_multilevel_mot_config()
    model = build_rate_equation_model()
    beams = build_multilevel_mot_beams(config=config)
    rows = []
    records = []
    payloads = [
        (selected, "surrogate", step, model, beams, config, default_anti_helmholtz_config())
        for step in time_steps_s
    ]
    with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
        results = list(executor.map(_simulate_trajectory_task, payloads))
    for step, (_, _, record) in zip(time_steps_s, results):
        print(f"[surrogate field study] timestep audit dt={step:.3e} s complete", flush=True)
        records.append(record)
        rows.append({"time_step_s": step, **classify_diagnostic_trajectory(record)})
    classifications_agree = len({row["trapped"] for row in rows}) == 1
    reasons_agree = len({row["classification"] for row in rows}) == 1
    final_positions = [np.asarray(record.positions_m[-1]) for record in records]
    final_velocities = [np.asarray(record.velocities_m_per_s[-1]) for record in records]
    payload = {
        "schema": "pmot.surrogate-field-timestep-audit.v1",
        "case": asdict(selected),
        "runs": rows,
        "capture_classifications_agree": classifications_agree,
        "classification_reasons_agree": reasons_agree,
        "final_position_difference_m": float(np.linalg.norm(final_positions[0] - final_positions[1])),
        "final_velocity_difference_m_per_s": float(np.linalg.norm(final_velocities[0] - final_velocities[1])),
    }
    path = _write_json(output_directory / "data" / "trajectories" / "timestep_audit.json", payload)
    return {"path": path, **payload}


def default_output_directory(project_root: Path) -> Path:
    return project_root / "outputs" / "diagnostics" / "pmot" / OUTPUT_DIRECTORY_NAME


__all__ = [
    "DiagnosticTrajectoryCase",
    "OUTPUT_DIRECTORY_NAME",
    "classify_diagnostic_trajectory",
    "create_field_diagnostics",
    "create_force_diagnostics",
    "default_diagnostic_trajectory_cases",
    "default_output_directory",
    "run_diagnostic_trajectory_panel",
    "run_timestep_audit",
]
