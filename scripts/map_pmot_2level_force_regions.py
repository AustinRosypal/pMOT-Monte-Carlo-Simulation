"""Map local restoring and damping regions of the effective two-level pMOT."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from pmot.configuration import RB87_MASS_KG
from pmot.mot_simple.configuration import default_simple_mot_config
from pmot.pMOT_2Level import build_pmot_2level_cooling_beams
from pmot.pMOT_2Level import build_pmot_2level_trapping_beams
from pmot.pMOT_2Level import evaluate_pmot_2level_local_observable
from pmot.pMOT_2Level import load_pmot_2level_polarizability_table


TRAPPING_POWER_W_PER_PATH = 38.294486e-3
MAP_EXTENT_M = 2.5e-3
MAP_SAMPLE_COUNT = 51
RADIAL_SAMPLE_COUNT = 151
RADIAL_VELOCITY_STEP_M_PER_S = 0.01
JACOBIAN_POSITION_STEP_M = 5.0e-6
OUTPUT_ROOT = Path(
    "outputs/diagnostics/pMOT_2Level/force_region_map_27mW_20261015"
)

CLASS_LABELS = {
    0: "neutral / unresolved",
    1: "restoring + damping",
    2: "restoring + anti-damping",
    3: "anti-restoring + damping",
    4: "anti-restoring + anti-damping",
}
CLASS_COLORS = ("#d4d4d8", "#15803d", "#d97706", "#7c3aed", "#dc2626")


def _force_sample(cooling, trapping, simple, table, position, velocity):
    return evaluate_pmot_2level_local_observable(
        cooling,
        trapping,
        position,
        velocity,
        path_powers_w=(TRAPPING_POWER_W_PER_PATH,) * 3,
        previous_axis=(0.0, 0.0, 1.0),
        simple_config=simple,
        polarizability_table=table,
    )


def _radial_observables(cooling, trapping, simple, table, position):
    position = np.asarray(position, dtype=float)
    radius = float(np.linalg.norm(position))
    radial_hat = position / radius if radius > 0.0 else np.asarray((1.0, 0.0, 0.0))
    center = _force_sample(
        cooling,
        trapping,
        simple,
        table,
        position,
        np.zeros(3),
    )
    force = np.asarray(center.force_n)
    radial_force = float(np.dot(force, radial_hat))
    plus = _force_sample(
        cooling,
        trapping,
        simple,
        table,
        position,
        RADIAL_VELOCITY_STEP_M_PER_S * radial_hat,
    )
    minus = _force_sample(
        cooling,
        trapping,
        simple,
        table,
        position,
        -RADIAL_VELOCITY_STEP_M_PER_S * radial_hat,
    )
    radial_velocity_slope = float(
        np.dot(np.asarray(plus.force_n) - np.asarray(minus.force_n), radial_hat)
        / (2.0 * RADIAL_VELOCITY_STEP_M_PER_S)
    )
    return center, force, radial_force, radial_velocity_slope


def _classify(radial_force, radial_velocity_slope, *, force_scale, slope_scale):
    force_tolerance = max(1.0e-28, 1.0e-4 * force_scale)
    slope_tolerance = max(1.0e-30, 1.0e-4 * slope_scale)
    restoring = radial_force < -force_tolerance
    anti_restoring = radial_force > force_tolerance
    damping = radial_velocity_slope < -slope_tolerance
    anti_damping = radial_velocity_slope > slope_tolerance
    result = np.zeros(np.shape(radial_force), dtype=np.int8)
    result[restoring & damping] = 1
    result[restoring & anti_damping] = 2
    result[anti_restoring & damping] = 3
    result[anti_restoring & anti_damping] = 4
    return result


def _map_plane(cooling, trapping, simple, table):
    coordinates = np.linspace(-MAP_EXTENT_M, MAP_EXTENT_M, MAP_SAMPLE_COUNT)
    shape = (MAP_SAMPLE_COUNT, MAP_SAMPLE_COUNT)
    force_x = np.empty(shape)
    force_z = np.empty(shape)
    radial_force = np.empty(shape)
    velocity_slope = np.empty(shape)
    intensity = np.empty(shape)
    for row, z_value in enumerate(coordinates):
        for column, x_value in enumerate(coordinates):
            center, force, radial, slope = _radial_observables(
                cooling,
                trapping,
                simple,
                table,
                (x_value, 0.0, z_value),
            )
            force_x[row, column] = force[0]
            force_z[row, column] = force[2]
            radial_force[row, column] = radial
            velocity_slope[row, column] = slope
            intensity[row, column] = sum(center.stark.component_intensities_w_per_m2)
    force_scale = float(np.max(np.abs(radial_force)))
    slope_scale = float(np.max(np.abs(velocity_slope)))
    classes = _classify(
        radial_force,
        velocity_slope,
        force_scale=force_scale,
        slope_scale=slope_scale,
    )
    center_index = MAP_SAMPLE_COUNT // 2
    classes[center_index, center_index] = 0
    return {
        "coordinates_m": coordinates,
        "force_x_n": force_x,
        "force_z_n": force_z,
        "radial_force_n": radial_force,
        "radial_velocity_slope_n_s_per_m": velocity_slope,
        "total_trapping_intensity_w_per_m2": intensity,
        "classes": classes,
    }


def _line_cut(cooling, trapping, simple, table, direction):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    radii = np.linspace(0.0, MAP_EXTENT_M, RADIAL_SAMPLE_COUNT)
    radial_force = np.empty_like(radii)
    velocity_slope = np.empty_like(radii)
    intensity = np.empty_like(radii)
    for index, radius in enumerate(radii):
        center, _, radial, slope = _radial_observables(
            cooling,
            trapping,
            simple,
            table,
            radius * direction,
        )
        radial_force[index] = radial
        velocity_slope[index] = slope
        intensity[index] = sum(center.stark.component_intensities_w_per_m2)
    classes = _classify(
        radial_force,
        velocity_slope,
        force_scale=float(np.max(np.abs(radial_force))),
        slope_scale=float(np.max(np.abs(velocity_slope))),
    )
    classes[0] = 0
    return {
        "radii_m": radii,
        "radial_force_n": radial_force,
        "radial_velocity_slope_n_s_per_m": velocity_slope,
        "total_trapping_intensity_w_per_m2": intensity,
        "classes": classes,
    }


def _intervals(radii, classes, selected_class=1):
    selected = np.asarray(classes) == selected_class
    intervals = []
    start = None
    for index, value in enumerate(selected):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(selected) - 1):
            end = index if value else index - 1
            intervals.append(
                [float(1.0e3 * radii[start]), float(1.0e3 * radii[end])]
            )
            start = None
    return intervals


def _full_jacobians(cooling, trapping, simple, table, position):
    position = np.asarray(position, dtype=float)
    position_jacobian = np.empty((3, 3))
    velocity_jacobian = np.empty((3, 3))
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = JACOBIAN_POSITION_STEP_M
        plus = np.asarray(
            _force_sample(
                cooling, trapping, simple, table, position + offset, np.zeros(3)
            ).force_n
        )
        minus = np.asarray(
            _force_sample(
                cooling, trapping, simple, table, position - offset, np.zeros(3)
            ).force_n
        )
        position_jacobian[:, axis] = (
            plus - minus
        ) / (2.0 * JACOBIAN_POSITION_STEP_M)
        velocity = np.zeros(3)
        velocity[axis] = RADIAL_VELOCITY_STEP_M_PER_S
        plus = np.asarray(
            _force_sample(cooling, trapping, simple, table, position, velocity).force_n
        )
        minus = np.asarray(
            _force_sample(cooling, trapping, simple, table, position, -velocity).force_n
        )
        velocity_jacobian[:, axis] = (
            plus - minus
        ) / (2.0 * RADIAL_VELOCITY_STEP_M_PER_S)
    return position_jacobian, velocity_jacobian


def _line_force_zeros(cooling, trapping, simple, table, direction, line_cut):
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    radii = line_cut["radii_m"]
    radial_force = line_cut["radial_force_n"]
    zeros = []
    for index in range(1, len(radii)):
        lower_force = radial_force[index - 1]
        upper_force = radial_force[index]
        if lower_force == 0.0 or lower_force * upper_force >= 0.0:
            continue
        fraction = abs(lower_force) / (abs(lower_force) + abs(upper_force))
        radius = float(radii[index - 1] + fraction * (radii[index] - radii[index - 1]))
        position = radius * direction
        observable = _force_sample(
            cooling,
            trapping,
            simple,
            table,
            position,
            np.zeros(3),
        )
        position_jacobian, velocity_jacobian = _full_jacobians(
            cooling,
            trapping,
            simple,
            table,
            position,
        )
        position_eigenvalues = np.linalg.eigvals(position_jacobian)
        velocity_eigenvalues = np.linalg.eigvals(velocity_jacobian)
        zeros.append(
            {
                "radius_mm": 1.0e3 * radius,
                "position_mm": (1.0e3 * position).tolist(),
                "force_magnitude_n": float(np.linalg.norm(observable.force_n)),
                "position_eigenvalues_n_per_m": [
                    [float(value.real), float(value.imag)]
                    for value in position_eigenvalues
                ],
                "velocity_eigenvalues_n_s_per_m": [
                    [float(value.real), float(value.imag)]
                    for value in velocity_eigenvalues
                ],
                "all_position_restoring": bool(
                    np.max(position_eigenvalues.real) < 0.0
                ),
                "all_velocity_damping": bool(
                    np.max(velocity_eigenvalues.real) < 0.0
                ),
            }
        )
    return zeros


def _gravity_equilibria(cooling, trapping, simple, table, positive_z_line_cut):
    radii = positive_z_line_cut["radii_m"]
    # By inversion symmetry F_z(0,0,-r) = -F_z(0,0,+r), while gravity is
    # constant and points toward -z.
    net_force_z = (
        -positive_z_line_cut["radial_force_n"]
        + RB87_MASS_KG * simple.gravity_acceleration_m_per_s2[2]
    )
    equilibria = []
    for index in range(1, len(radii)):
        lower_force = net_force_z[index - 1]
        upper_force = net_force_z[index]
        if lower_force == 0.0 or lower_force * upper_force >= 0.0:
            continue
        fraction = abs(lower_force) / (abs(lower_force) + abs(upper_force))
        radius = float(radii[index - 1] + fraction * (radii[index] - radii[index - 1]))
        position = np.asarray((0.0, 0.0, -radius))
        observable = _force_sample(
            cooling,
            trapping,
            simple,
            table,
            position,
            np.zeros(3),
        )
        net_force = np.asarray(observable.force_n) + RB87_MASS_KG * np.asarray(
            simple.gravity_acceleration_m_per_s2
        )
        position_jacobian, velocity_jacobian = _full_jacobians(
            cooling,
            trapping,
            simple,
            table,
            position,
        )
        position_eigenvalues = np.linalg.eigvals(position_jacobian)
        velocity_eigenvalues = np.linalg.eigvals(velocity_jacobian)
        equilibria.append(
            {
                "position_mm": (1.0e3 * position).tolist(),
                "net_force_magnitude_n": float(np.linalg.norm(net_force)),
                "position_eigenvalues_n_per_m": [
                    [float(value.real), float(value.imag)]
                    for value in position_eigenvalues
                ],
                "velocity_eigenvalues_n_s_per_m": [
                    [float(value.real), float(value.imag)]
                    for value in velocity_eigenvalues
                ],
                "all_position_restoring": bool(
                    np.max(position_eigenvalues.real) < 0.0
                ),
                "all_velocity_damping": bool(
                    np.max(velocity_eigenvalues.real) < 0.0
                ),
                "dynamically_stable_linearization": bool(
                    np.max(position_eigenvalues.real) < 0.0
                    and np.max(velocity_eigenvalues.real) < 0.0
                ),
            }
        )
    return equilibria


def _plane_fractions(plane):
    x, z = np.meshgrid(plane["coordinates_m"], plane["coordinates_m"])
    inside_core = x**2 + z**2 <= (2.0e-3) ** 2
    denominator = int(np.sum(inside_core))
    return {
        CLASS_LABELS[value]: float(
            np.sum((plane["classes"] == value) & inside_core) / denominator
        )
        for value in CLASS_LABELS
    }


def _plot(results, path):
    figure = plt.figure(figsize=(13.0, 11.2), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=(0.13, 1.0, 1.0))
    header = figure.add_subplot(grid[0, :])
    header.axis("off")
    axes = np.asarray(
        [
            [figure.add_subplot(grid[1, 0]), figure.add_subplot(grid[1, 1])],
            [figure.add_subplot(grid[2, 0]), figure.add_subplot(grid[2, 1])],
        ]
    )
    colormap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), colormap.N)
    for axis, key, title in (
        (axes[0, 0], "canonical_ppm", r"Canonical trapping helicity $++-$"),
        (axes[0, 1], "reversed_mmp", r"Reversed trapping helicity $--+$"),
    ):
        plane = results[key]["plane"]
        coordinates_mm = 1.0e3 * plane["coordinates_m"]
        axis.imshow(
            plane["classes"],
            origin="lower",
            extent=(coordinates_mm[0], coordinates_mm[-1], coordinates_mm[0], coordinates_mm[-1]),
            cmap=colormap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        intensity = np.log10(plane["total_trapping_intensity_w_per_m2"])
        contour_levels = np.linspace(float(np.min(intensity)), float(np.max(intensity)), 5)[1:-1]
        axis.contour(
            coordinates_mm,
            coordinates_mm,
            intensity,
            levels=contour_levels,
            colors="black",
            linewidths=0.55,
            alpha=0.45,
        )
        stride = 5
        force_x = plane["force_x_n"][::stride, ::stride]
        force_z = plane["force_z_n"][::stride, ::stride]
        magnitude = np.hypot(force_x, force_z)
        normalized_x = np.divide(force_x, magnitude, out=np.zeros_like(force_x), where=magnitude > 0.0)
        normalized_z = np.divide(force_z, magnitude, out=np.zeros_like(force_z), where=magnitude > 0.0)
        sampled = coordinates_mm[::stride]
        axis.quiver(
            sampled,
            sampled,
            normalized_x,
            normalized_z,
            color="black",
            alpha=0.7,
            pivot="middle",
            scale=24.0,
            width=0.004,
        )
        axis.add_patch(
            Circle((0.0, 0.0), 2.0, fill=False, color="black", linestyle="--", linewidth=1.0)
        )
        axis.plot(0.0, 0.0, marker="x", color="black", markersize=6)
        for equilibrium in results[key]["gravity_equilibria"]:
            stable = equilibrium["dynamically_stable_linearization"]
            axis.plot(
                equilibrium["position_mm"][0],
                equilibrium["position_mm"][2],
                marker="*",
                markerfacecolor="white" if not stable else "#facc15",
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=12,
            )
        axis.set_title(title)
        axis.set_xlabel("x [mm]")
        axis.set_ylabel("z [mm]")
        axis.set_xlim(-2.5, 2.5)
        axis.set_ylim(-2.5, 2.5)
        axis.grid(False)

    canonical = results["canonical_ppm"]["line_cuts"]
    line_colors = {"x axis": "#2563eb", "z axis": "#dc2626", "body diagonal": "#111827"}
    line_styles = {"x axis": "--", "z axis": "-", "body diagonal": ":"}
    for label, line in canonical.items():
        radius_mm = 1.0e3 * line["radii_m"]
        axes[1, 0].plot(
            radius_mm,
            1.0e24 * line["radial_force_n"],
            label=label,
            color=line_colors[label],
            linestyle=line_styles[label],
            linewidth=1.8,
        )
        axes[1, 1].plot(
            radius_mm,
            1.0e24 * line["radial_velocity_slope_n_s_per_m"],
            label=label,
            color=line_colors[label],
            linestyle=line_styles[label],
            linewidth=1.8,
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title(r"Canonical $++-$: radial force")
    axes[1, 0].set_xlabel("distance from origin [mm]")
    axes[1, 0].set_ylabel(r"$F_r$ [$10^{-24}$ N]")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title(r"Canonical $++-$: radial velocity slope")
    axes[1, 1].set_xlabel("distance from origin [mm]")
    axes[1, 1].set_ylabel(r"$\partial F_r/\partial v_r$ [$10^{-24}$ N s/m]")
    for axis in axes[1]:
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
        axis.set_xlim(0.0, 2.5)

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=CLASS_COLORS[value],
            markeredgecolor="none",
            markersize=9,
            label=CLASS_LABELS[value],
        )
        for value in range(1, 5)
    ]
    class_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markerfacecolor="#facc15",
            markeredgecolor="black",
            markersize=11,
            label="gravity-balanced stable equilibrium",
        )
    )
    header.text(
        0.5,
        0.98,
        "Effective two-level pMOT force regions: 27 mW cooling, physical Stark sign",
        ha="center",
        va="top",
        fontsize=14,
        transform=header.transAxes,
    )
    header.legend(
        handles=class_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    output_root = OUTPUT_ROOT.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    simple = default_simple_mot_config()
    cooling = build_pmot_2level_cooling_beams(simple_config=simple)
    table = load_pmot_2level_polarizability_table()
    configurations = {
        "canonical_ppm": ("sigma+", "sigma+", "sigma-"),
        "reversed_mmp": ("sigma-", "sigma-", "sigma+"),
    }
    directions = {
        "x axis": (1.0, 0.0, 0.0),
        "z axis": (0.0, 0.0, 1.0),
        "body diagonal": (1.0, 1.0, 1.0),
    }
    results = {}
    manifest = {
        "model": "effective F=2 to F'=3 provisional diagnostic",
        "ac_stark_definition": "DeltaE_AC = U_e - U_g",
        "effective_detuning": "Delta0 - k.v/(2*pi) - DeltaE_AC/h",
        "cooling_power_w_per_beam": cooling[0].intensity_beam.power_w,
        "cooling_detuning_hz": simple.cooling_detuning_hz,
        "trapping_power_w_per_path": TRAPPING_POWER_W_PER_PATH,
        "trapping_power_status": "historical 20 G/cm proxy; not a physical default",
        "map_plane": "y=0 x-z plane",
        "map_extent_m": MAP_EXTENT_M,
        "map_sample_count_per_axis": MAP_SAMPLE_COUNT,
        "classification_definition": {
            "restoring": "dot(F(r,0), r_hat) < 0",
            "anti_restoring": "dot(F(r,0), r_hat) > 0",
            "damping": "d dot(F,r_hat) / d dot(v,r_hat) < 0",
            "locally_containing": "radially restoring and radially damping; not a full trapping proof",
        },
        "configurations": {},
    }
    for key, helicities in configurations.items():
        trapping = build_pmot_2level_trapping_beams(
            incident_helicities_by_axis=helicities,
            retro_helicities_by_axis=helicities,
        )
        plane = _map_plane(cooling, trapping, simple, table)
        line_cuts = {
            label: _line_cut(cooling, trapping, simple, table, direction)
            for label, direction in directions.items()
        }
        gravity_equilibria = _gravity_equilibria(
            cooling,
            trapping,
            simple,
            table,
            line_cuts["z axis"],
        )
        results[key] = {
            "plane": plane,
            "line_cuts": line_cuts,
            "gravity_equilibria": gravity_equilibria,
        }
        manifest["configurations"][key] = {
            "helicities_x_y_z": list(helicities),
            "core_plane_area_fractions": _plane_fractions(plane),
            "locally_containing_intervals_mm": {
                label: _intervals(line["radii_m"], line["classes"])
                for label, line in line_cuts.items()
            },
            "positive_radius_force_zeros": {
                label: _line_force_zeros(
                    cooling,
                    trapping,
                    simple,
                    table,
                    directions[label],
                    line,
                )
                for label, line in line_cuts.items()
            },
            "gravity_equilibria_on_negative_z": gravity_equilibria,
        }
        np.savez_compressed(
            output_root / f"{key}_map.npz",
            **{f"plane_{name}": value for name, value in plane.items()},
            **{
                f"{label.replace(' ', '_')}_{name}": value
                for label, line in line_cuts.items()
                for name, value in line.items()
            },
        )

    figure_path = output_root / "force_regions.png"
    _plot(results, figure_path)
    manifest["figure_path"] = str(figure_path)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved {figure_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
