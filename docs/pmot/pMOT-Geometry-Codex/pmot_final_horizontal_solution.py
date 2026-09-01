"""Final horizontal-path pMOT geometry solution.

This script is the current entry point for the horizontal paths. It uses the
45 degree glass-cell geometry and AC508-080-C lenses, with no coil exclusion
zone. The only modeled hard object in the horizontal path is the glass cell
envelope itself.
"""

from __future__ import annotations

from math import cos, pi, sin, tan
from pathlib import Path

from pmot_round_trip_geometry import (
    GeometryConfig,
    LensSpec,
    RoundTripGeometry,
    print_780_summary,
    print_summary,
    solve_780_cat_eye,
)


def ac508_080_c_lens() -> LensSpec:
    return LensSpec(
        name="AC508-080-C",
        focal_length=80.3e-3,
        efl=80.3e-3,
        bfl=66.9e-3,
        center_thickness=20.5e-3,
        mount_length=27.7e-3,
        mount_overhang=3.9e-3,
        mount_half_height=33e-3,
    )


def solve_horizontal_geometry() -> tuple[RoundTripGeometry, dict[str, object], dict[str, object]]:
    lens = ac508_080_c_lens()
    config = GeometryConfig(
        cell_aoi_deg=45.0,
        cell_outer_diameter=30e-3,
        cell_wall_thickness=5e-3,
        focus_separation=20e-3,
        mirror_gap=40e-3,
        input_radius=17.5e-3,
        samples_per_space=500,
    )
    geometry = RoundTripGeometry(lens, config)
    solution = geometry.solve()
    beam_780 = solve_780_cat_eye(geometry, solution, mot_beam_diameter=12.7e-3)
    return geometry, solution, beam_780


def print_horizontal_mechanics(geometry: RoundTripGeometry, solution: dict[str, object]) -> None:
    print()
    print("Horizontal mechanical envelopes")
    print("-------------------------------")
    print("No coil exclusion is applied for the horizontal paths.")
    print(f"L1 mount outer left              = {1e3 * solution['z_lens1_mount_left']:.3f} mm")
    print(f"L1 mount outer right             = {1e3 * solution['z_lens1_mount_right']:.3f} mm")
    print(f"L2 mount outer left              = {1e3 * solution['z_lens2_mount_left']:.3f} mm")
    print(f"L2 mount outer right             = {1e3 * solution['z_lens2_mount_right']:.3f} mm")
    print(f"Left glass-cell outer edge       = {1e3 * geometry.z_outer_left:.3f} mm")
    print(f"Right glass-cell outer edge      = {1e3 * geometry.z_outer_right:.3f} mm")
    print(f"L1-to-left-cell clearance        = {1e3 * (geometry.z_outer_left - solution['z_lens1_mount_right']):.3f} mm")
    print(f"L2-to-right-cell clearance       = {1e3 * (solution['z_lens2_mount_left'] - geometry.z_outer_right):.3f} mm")


def horizontal_cell_to_plot(geometry: RoundTripGeometry, u_value: float, v_value: float) -> tuple[float, float]:
    """Map the simplified horizontal-cell drawing coordinates into plot z/y."""
    theta_i = geometry.theta_i
    normal_angle = theta_i
    cell_axis_angle = theta_i + pi / 2
    z = u_value * cos(cell_axis_angle) + v_value * cos(normal_angle)
    y = u_value * sin(cell_axis_angle) + v_value * sin(normal_angle)
    return z, y


def horizontal_rect_points(
    geometry: RoundTripGeometry,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
) -> list[tuple[float, float]]:
    return [
        tuple(1e3 * value for value in horizontal_cell_to_plot(geometry, u, v))
        for u, v in [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
    ]


def draw_horizontal_cell_rect(ax, geometry: RoundTripGeometry, u0: float, u1: float, v0: float, v1: float, **kwargs) -> None:
    points = horizontal_rect_points(geometry, u0, u1, v0, v1)
    ax.fill([point[0] for point in points], [point[1] for point in points], **kwargs)


def draw_horizontal_cell_outline(ax, geometry: RoundTripGeometry, u0: float, u1: float, v0: float, v1: float, **kwargs) -> None:
    points = horizontal_rect_points(geometry, u0, u1, v0, v1)
    closed = points + [points[0]]
    ax.plot([point[0] for point in closed], [point[1] for point in closed], **kwargs)


def draw_simplified_horizontal_cell(ax, geometry: RoundTripGeometry) -> None:
    """Draw the simplified 100 mm cell body and open-end collar."""
    cell_body_length = 100e-3
    cell_outer_height = 30e-3
    side_wall = 5e-3
    left_end_glass = 5e-3
    pedestal_depth = 5e-3
    pedestal_total_extra_height = 10e-3
    pedestal_half_extra = pedestal_total_extra_height / 2

    u_min = -cell_body_length / 2
    u_max = cell_body_length / 2
    v_min = -cell_outer_height / 2
    v_max = cell_outer_height / 2

    u_inner_min = u_min
    u_inner_max = u_max - left_end_glass
    v_inner_min = v_min + side_wall
    v_inner_max = v_max - side_wall

    u_col_min = u_min
    u_col_max = u_min + pedestal_depth
    v_col_min = v_min - pedestal_half_extra
    v_col_max = v_max + pedestal_half_extra

    draw_horizontal_cell_rect(
        ax,
        geometry,
        u_min,
        u_max,
        v_min,
        v_max,
        facecolor="orange",
        edgecolor="orange",
        alpha=0.12,
        linewidth=1.6,
        zorder=0.5,
    )
    draw_horizontal_cell_rect(
        ax,
        geometry,
        u_inner_min,
        u_inner_max,
        v_inner_min,
        v_inner_max,
        facecolor="white",
        edgecolor="orange",
        alpha=1.0,
        linewidth=1.2,
        zorder=0.7,
    )
    draw_horizontal_cell_rect(
        ax,
        geometry,
        u_col_min,
        u_col_max,
        v_col_min,
        v_col_max,
        facecolor="orange",
        edgecolor="orange",
        alpha=0.22,
        linewidth=1.8,
        zorder=0.9,
    )
    draw_horizontal_cell_rect(
        ax,
        geometry,
        u_col_min,
        u_col_max,
        v_inner_min,
        v_inner_max,
        facecolor="white",
        edgecolor="orange",
        alpha=1.0,
        linewidth=1.2,
        zorder=1.0,
    )

    for rect, linewidth, zorder in [
        ((u_min, u_max, v_min, v_max), 1.8, 1.2),
        ((u_inner_min, u_inner_max, v_inner_min, v_inner_max), 1.2, 1.3),
        ((u_col_min, u_col_max, v_col_min, v_col_max), 1.8, 1.4),
        ((u_col_min, u_col_max, v_inner_min, v_inner_max), 1.2, 1.5),
    ]:
        draw_horizontal_cell_outline(
            ax,
            geometry,
            *rect,
            color="orange",
            linewidth=linewidth,
            zorder=zorder,
        )


def draw_horizontal_cell_outer_size_guides(ax, geometry: RoundTripGeometry, extra_u: float = 35e-3) -> None:
    cell_body_length = 100e-3
    cell_outer_height = 30e-3
    theta_i = geometry.theta_i
    normal_angle = theta_i
    cell_axis_angle = theta_i + pi / 2
    samples = 400
    u_values = [
        -cell_body_length / 2 - extra_u + i * (cell_body_length + 2 * extra_u) / (samples - 1)
        for i in range(samples)
    ]

    for v in [-cell_outer_height / 2 - 5e-3, cell_outer_height / 2 + 5e-3]:
        z_values = [
            1e3 * (u * cos(cell_axis_angle) + v * cos(normal_angle))
            for u in u_values
        ]
        y_values = [
            1e3 * (u * sin(cell_axis_angle) + v * sin(normal_angle))
            for u in u_values
        ]
        ax.plot(
            z_values,
            y_values,
            linestyle="--",
            dashes=(8, 5),
            color="red",
            linewidth=1.6,
            alpha=0.9,
            zorder=0,
        )


def draw_raytrace_interfaces(ax, geometry: RoundTripGeometry, y_half: float = 60e-3) -> None:
    samples = 300
    y_values = [-y_half + i * (2 * y_half) / (samples - 1) for i in range(samples)]
    for z_center in [
        geometry.z_outer_left,
        geometry.z_inner_left,
        geometry.z_inner_right,
        geometry.z_outer_right,
    ]:
        ax.plot(
            [1e3 * (z_center - y * tan(geometry.theta_i)) for y in y_values],
            [1e3 * y for y in y_values],
            color="purple",
            linestyle=":",
            linewidth=1.4,
            alpha=0.8,
            zorder=8,
        )


def draw_final_horizontal_png(
    geometry: RoundTripGeometry,
    solution: dict[str, object],
    output_path: Path,
    beam_780: dict[str, object] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)

    z_forward = solution["z_forward"]
    w_forward = solution["forward"]["w"]
    z_reverse = solution["z_reverse"]
    w_reverse = solution["reverse"]["w"]

    ax.plot([1e3 * z for z in z_forward], [1e3 * w for w in w_forward], color="C0", linewidth=1.8, label="forward beam")
    ax.plot([1e3 * z for z in z_forward], [-1e3 * w for w in w_forward], color="C0", linewidth=1.8)
    ax.plot([1e3 * z for z in z_reverse], [1e3 * w for w in w_reverse], color="C1", linewidth=1.8, linestyle="--", label="return beam")
    ax.plot([1e3 * z for z in z_reverse], [-1e3 * w for w in w_reverse], color="C1", linewidth=1.8, linestyle="--")

    if beam_780 is not None:
        z_780 = [*beam_780["z_left"], *beam_780["z_right"]]
        w_780 = [*beam_780["w_left"], *beam_780["w_right"]]
        order = sorted(range(len(z_780)), key=lambda idx: z_780[idx])
        z_sorted = [1e3 * z_780[idx] for idx in order]
        w_sorted = [1e3 * w_780[idx] for idx in order]
        ax.plot(z_sorted, w_sorted, color="#2ca02c", linewidth=2.2, label="780 incident")
        ax.plot(z_sorted, [-w for w in w_sorted], color="#2ca02c", linewidth=2.2)
        ax.plot(z_sorted, w_sorted, color="#9467bd", linewidth=1.6, linestyle=(0, (6, 4)), label="780 retro")
        ax.plot(z_sorted, [-w for w in w_sorted], color="#9467bd", linewidth=1.6, linestyle=(0, (6, 4)))

    draw_simplified_horizontal_cell(ax, geometry)
    draw_horizontal_cell_outer_size_guides(ax, geometry)
    draw_raytrace_interfaces(ax, geometry)

    for left_key, right_key, label in [
        ("z_lens1_mount_left", "z_lens1_mount_right", "L1 mounted lens"),
        ("z_lens2_mount_left", "z_lens2_mount_right", "L2 mounted lens"),
    ]:
        z_left = 1e3 * solution[left_key]
        z_right = 1e3 * solution[right_key]
        y_mount = 1e3 * geometry.lens.mount_half_height
        ax.add_patch(
            Rectangle(
                (z_left, -y_mount),
                z_right - z_left,
                2 * y_mount,
                facecolor="0.25",
                edgecolor="0.25",
                linewidth=1.5,
                alpha=0.10,
            )
        )
        ax.text(0.5 * (z_left + z_right), 0.88 * y_mount, label, ha="center", va="top", fontsize=10)

    for key in ["z_lens1", "z_lens2"]:
        ax.axvline(1e3 * solution[key], color="black", linewidth=2.0)
    for key in ["z_lens1_other", "z_lens1_bfl", "z_lens2_bfl", "z_lens2_other"]:
        ax.axvline(1e3 * solution[key], color="black", linestyle="--", linewidth=1.0)

    ax.axvline(1e3 * solution["z_mirror"], color="gray", linewidth=2.0)
    ax.axvline(0.0, color="0.5", linewidth=1.0, linestyle=":")
    ax.axvline(1e3 * solution["z_focus1"], color="C0", linestyle=":", linewidth=1.5)
    ax.axvline(1e3 * solution["z_focus2"], color="C1", linestyle=":", linewidth=1.5)
    ax.plot(1e3 * solution["z_focus1"], 0.0, marker="o", color="C0")
    ax.plot(1e3 * solution["z_focus2"], 0.0, marker="o", color="C1")

    if beam_780 is not None:
        ax.axvline(1e3 * beam_780["left_focus_z"], color="#2ca02c", linestyle="-.", linewidth=1.8)
        ax.axvline(1e3 * beam_780["right_focus_z"], color="#2ca02c", linestyle=":", linewidth=1.6)
        ax.plot(1e3 * beam_780["left_focus_z"], 0.0, marker="D", color="#2ca02c", markersize=5)
        ax.plot(1e3 * beam_780["right_focus_z"], 0.0, marker="D", color="#2ca02c", markersize=5)
        ax.text(1e3 * beam_780["left_focus_z"], -42, "780 cat-eye\nfilter", color="#2ca02c", ha="center", va="center")
        ax.text(1e3 * beam_780["right_focus_z"], 42, "780 input\nfocus", color="#2ca02c", ha="center", va="center")

    label_y = 60
    ax.text(1e3 * solution["z_lens1"], label_y, f"Lens 1\n{geometry.lens.name}", ha="center", va="bottom")
    ax.text(1e3 * solution["z_lens2"], label_y, "Lens 2", ha="center", va="bottom")
    ax.text(1e3 * solution["z_mirror"], label_y, "Mirror\n0 deg AOI", ha="center", va="bottom")
    ax.text(0.0, 55, "z = 0\ncell center", ha="center", va="top")
    ax.text(1e3 * solution["z_focus1"], -55, f"focus 1\n{1e3 * solution['z_focus1']:.2f} mm", color="C0", ha="center", va="top")
    ax.text(1e3 * solution["z_focus2"], -55, f"focus 2\n{1e3 * solution['z_focus2']:.2f} mm", color="C1", ha="center", va="top")

    ax.set_xlabel("z along beam axis, relative to cell center [mm]")
    ax.set_ylabel("beam radius / envelope [mm]")
    if beam_780 is None:
        title = f"1530 nm round-trip Gaussian beam tracing; cell AOI = {geometry.config.cell_aoi_deg:.1f} deg"
    else:
        title = f"1530 nm round-trip plus 780 nm cat-eye overlay; cell AOI = {geometry.config.cell_aoi_deg:.1f} deg"
    ax.set_title(title, pad=18)
    ax.set_xlim(-205 if beam_780 is not None else -157, 190 if beam_780 is not None else 135)
    ax.set_ylim(-60, 60)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    geometry, solution, beam_780 = solve_horizontal_geometry()
    output_png = Path(__file__).with_name("FinalHorizontalSolution.png")
    output_780_png = Path(__file__).with_name("FinalHorizontalSolution_with_780.png")

    draw_final_horizontal_png(geometry, solution, output_png)
    draw_final_horizontal_png(geometry, solution, output_780_png, beam_780=beam_780)

    print_summary(geometry, solution)
    print_horizontal_mechanics(geometry, solution)
    print_780_summary(beam_780)
    print()
    print(f"Saved PNG: {output_png}")
    print(f"Saved 780 overlay PNG: {output_780_png}")


if __name__ == "__main__":
    main()
