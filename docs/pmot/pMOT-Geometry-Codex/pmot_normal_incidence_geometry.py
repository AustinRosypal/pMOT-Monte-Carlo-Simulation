"""1530 nm normal-incidence pMOT geometry with AC254-040-C lenses.

This is intentionally separate from pmot_round_trip_geometry.py. It focuses on
the normal-incidence 1530 nm path only:

* Cell length along this beam axis is 40 mm with 5 mm walls.
* Lens 1 and Lens 2 use AC254-040-C parameters.
* The 1530 beam uses a 12 degree one-sided cone at the 40 mm lens.
* Lens 1 is solved to put the first focus at -10 mm.
* Lens 2 is moved left to test a 38 mm right-angle cube before the 1530 mirror
  while preserving the +10 mm return focus.
"""

from __future__ import annotations

from math import pi, tan
from pathlib import Path

from pmot_round_trip_geometry import (
    GeometryConfig,
    LensSpec,
    RoundTripGeometry,
    propagate_q,
    q_from_radius_and_curvature,
    waist_in_window,
)

STRAIGHT_SECTION_LIMIT_MM = 89.905
COIL_EXCLUSION_MM = 38.0
CUBE_SIZE_MM = 38.0
MOT_WAVELENGTH = 780e-9
MOT_INPUT_DIAMETER = 11.0e-3


def ac254_040_c_lens() -> LensSpec:
    return LensSpec(
        name="AC254-040-C",
        focal_length=40.0e-3,
        efl=40.0e-3,
        bfl=32.8e-3,
        center_thickness=15.5e-3,
        mount_length=15.5e-3,
        mount_overhang=0.0,
        mount_half_height=18e-3,
    )


def round_trip_with_positions(
    geometry: RoundTripGeometry,
    z_lens1: float,
    z_lens2: float,
    z_mirror: float,
) -> dict[str, object]:
    forward = propagate_q(
        geometry.q_in,
        geometry.build_forward_system(z_lens1, z_lens2, z_mirror),
        geometry.config.wavelength,
        geometry.config.samples_per_space,
    )
    reverse = propagate_q(
        forward["q_final"],
        geometry.build_reverse_system(z_lens1, z_lens2, z_mirror),
        geometry.config.wavelength,
        geometry.config.samples_per_space,
    )
    z_forward = [z_lens1 + s for s in forward["s"]]
    z_reverse = [z_mirror - s for s in reverse["s"]]
    z_focus1, w_focus1 = waist_in_window(z_forward, forward["w"], -0.030, 0.030)
    z_focus2, w_focus2 = waist_in_window(z_reverse, reverse["w"], -0.030, 0.030)
    return {
        "forward": forward,
        "reverse": reverse,
        "z_forward": z_forward,
        "z_reverse": z_reverse,
        "z_focus1": z_focus1,
        "z_focus2": z_focus2,
        "w_focus1": w_focus1,
        "w_focus2": w_focus2,
    }


def solve_mirror_for_return_focus(
    geometry: RoundTripGeometry,
    z_lens1: float,
    z_lens2: float,
    z_target: float = 10e-3,
    passes: int = 4,
    points: int = 81,
    z_min: float | None = None,
) -> float:
    low_bound = z_lens2 + 2e-3 if z_min is None else z_min
    low = low_bound
    high = 0.160
    best = high
    for _ in range(passes):
        grid = [low + i * (high - low) / (points - 1) for i in range(points)]
        scored = []
        for z_mirror in grid:
            try:
                result = round_trip_with_positions(geometry, z_lens1, z_lens2, z_mirror)
            except ValueError:
                continue
            scored.append((abs(result["z_focus2"] - z_target), z_mirror))
        _, best = min(scored, key=lambda item: item[0])
        step = grid[1] - grid[0]
        low = max(low_bound, best - 6 * step)
        high = best + 6 * step
    return best


def add_lens_surfaces(solution: dict[str, object], lens: LensSpec) -> None:
    principal_to_bfl = lens.efl - lens.bfl
    z_lens1 = solution["z_lens1"]
    z_lens2 = solution["z_lens2"]
    solution["z_lens1_bfl_surface"] = z_lens1 + principal_to_bfl
    solution["z_lens1_outer_surface"] = solution["z_lens1_bfl_surface"] - lens.center_thickness
    solution["z_lens2_cell_side_bfl_surface"] = z_lens2 - principal_to_bfl
    solution["z_lens2_outer_surface"] = solution["z_lens2_cell_side_bfl_surface"] + lens.center_thickness


def trace_780_mot_path(
    geometry: RoundTripGeometry,
    solution: dict[str, object],
    z_start: float = 0.135,
) -> dict[str, object]:
    lens = geometry.lens
    z_lens1 = solution["z_lens1"]
    z_lens2 = solution["z_lens2"]
    z_relay1 = z_lens2 + 2 * lens.focal_length

    if z_start <= z_relay1:
        raise ValueError("780 start point must be to the right of the first 4F lens.")

    q0 = q_from_radius_and_curvature(MOT_INPUT_DIAMETER / 2, float("inf"), MOT_WAVELENGTH)
    system = [
        {"type": "space", "d_plot": z_start - z_relay1, "d_abcd": z_start - z_relay1},
        {"type": "lens", "f": lens.focal_length},
        {"type": "space", "d_plot": z_relay1 - z_lens2, "d_abcd": z_relay1 - z_lens2},
        {"type": "lens", "f": lens.focal_length},
        {"type": "space", "d_plot": z_lens2 - geometry.z_outer_right, "d_abcd": z_lens2 - geometry.z_outer_right},
        {"type": "slab", "d_plot": geometry.wall_length_plot, "d_abcd": geometry.wall_b_eff},
        {"type": "space", "d_plot": geometry.vacuum_length_beam, "d_abcd": geometry.vacuum_length_beam},
        {"type": "slab", "d_plot": geometry.wall_length_plot, "d_abcd": geometry.wall_b_eff},
        {"type": "space", "d_plot": geometry.z_outer_left - z_lens1, "d_abcd": geometry.z_outer_left - z_lens1},
        {"type": "lens", "f": lens.focal_length},
        {"type": "space", "d_plot": 0.075, "d_abcd": 0.075},
    ]
    result = propagate_q(q0, system, MOT_WAVELENGTH, geometry.config.samples_per_space)
    z_global = [z_start - s for s in result["s"]]
    z_focus, w_focus = waist_in_window(z_global, result["w"], -0.130, -0.070)

    return {
        "trace": result,
        "z": z_global,
        "z_start": z_start,
        "z_relay1": z_relay1,
        "z_relay_focus": z_lens2 + lens.focal_length,
        "z_focus": z_focus,
        "w_focus": w_focus,
        "mot_diameter": MOT_INPUT_DIAMETER,
    }


def solve_compact_lens2_and_mirror(
    geometry: RoundTripGeometry,
    z_lens1: float,
    min_mirror_clearance: float,
    min_l2_cell_clearance: float = 2e-3,
) -> dict[str, object]:
    """Find the closest mirror layout that keeps L2 and the mirror separated."""

    lens = geometry.lens
    principal_to_bfl = lens.efl - lens.bfl
    z_lens2_min = max(
        geometry.z_outer_right + min_l2_cell_clearance + principal_to_bfl,
        42.8e-3,
    )
    z_lens2_max = 43.5e-3

    best: dict[str, object] | None = None
    for i in range(36):
        z_lens2 = z_lens2_min + i * (z_lens2_max - z_lens2_min) / 35
        z_mirror = solve_mirror_for_return_focus(geometry, z_lens1, z_lens2, passes=4, points=61)
        result = round_trip_with_positions(geometry, z_lens1, z_lens2, z_mirror)
        candidate = {
            **result,
            "z_lens1": z_lens1,
            "z_lens2": z_lens2,
            "z_mirror": z_mirror,
        }
        add_lens_surfaces(candidate, lens)

        focus_error = abs(candidate["z_focus2"] - 10e-3)
        mirror_clearance = candidate["z_mirror"] - candidate["z_lens2_outer_surface"]
        if focus_error > 0.10e-3:
            continue
        if mirror_clearance < min_mirror_clearance:
            continue

        if best is None or candidate["z_mirror"] < best["z_mirror"]:
            best = candidate

    if best is None:
        raise RuntimeError("No compact Lens 2 / mirror solution found with the requested clearance.")

    z_lens2 = best["z_lens2"]
    z_mirror = solve_mirror_for_return_focus(geometry, z_lens1, z_lens2)
    result = round_trip_with_positions(geometry, z_lens1, z_lens2, z_mirror)
    refined = {
        **result,
        "z_lens1": z_lens1,
        "z_lens2": z_lens2,
        "z_mirror": z_mirror,
        "min_mirror_clearance": min_mirror_clearance,
        "min_l2_cell_clearance": min_l2_cell_clearance,
    }
    add_lens_surfaces(refined, lens)
    return refined


def draw_1530_normal_png(
    geometry: RoundTripGeometry,
    solution: dict[str, object],
    mot_solution: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    straight_limit_mm = STRAIGHT_SECTION_LIMIT_MM
    cube_half_height_mm = CUBE_SIZE_MM / 2
    coil_exclusion_mm = COIL_EXCLUSION_MM

    z_forward = solution["z_forward"]
    z_reverse = solution["z_reverse"]
    w_forward = solution["forward"]["w"]
    w_reverse = solution["reverse"]["w"]

    ax.plot([1e3 * z for z in z_forward], [1e3 * w for w in w_forward], color="C0", linewidth=2.0, label="1530 forward")
    ax.plot([1e3 * z for z in z_forward], [-1e3 * w for w in w_forward], color="C0", linewidth=2.0)
    ax.plot([1e3 * z for z in z_reverse], [1e3 * w for w in w_reverse], color="C1", linewidth=2.0, linestyle="--", label="1530 return")
    ax.plot([1e3 * z for z in z_reverse], [-1e3 * w for w in w_reverse], color="C1", linewidth=2.0, linestyle="--")

    z_mot = mot_solution["z"]
    w_mot = mot_solution["trace"]["w"]
    ax.plot([1e3 * z for z in z_mot], [1e3 * w for w in w_mot], color="C2", linewidth=2.0, label="780 incident")
    ax.plot([1e3 * z for z in z_mot], [-1e3 * w for w in w_mot], color="C2", linewidth=2.0)

    ax.axvspan(-coil_exclusion_mm, coil_exclusion_mm, color="red", alpha=0.07, zorder=0)
    for z_limit in [-coil_exclusion_mm, coil_exclusion_mm]:
        ax.axvline(z_limit, color="red", linewidth=1.5, linestyle=":")
    ax.text(
        0,
        43,
        "coil exclusion\nno optics for |z| < 38 mm",
        color="red",
        ha="center",
        va="top",
        fontsize=9,
    )

    glass_color = "orange"
    y_cell_half_mm = 42
    for z0, z1 in [
        (geometry.z_outer_left, geometry.z_inner_left),
        (geometry.z_inner_right, geometry.z_outer_right),
    ]:
        ax.add_patch(
            Rectangle(
                (1e3 * z0, -y_cell_half_mm),
                1e3 * (z1 - z0),
                2 * y_cell_half_mm,
                facecolor=glass_color,
                edgecolor=glass_color,
                linewidth=1.4,
                alpha=0.14,
            )
        )
    ax.add_patch(
        Rectangle(
            (1e3 * geometry.z_inner_left, -y_cell_half_mm),
            1e3 * (geometry.z_inner_right - geometry.z_inner_left),
            2 * y_cell_half_mm,
            facecolor="none",
            edgecolor=glass_color,
            linewidth=1.2,
            linestyle=":",
            alpha=0.85,
        )
    )

    for key in ["z_lens1", "z_lens2"]:
        ax.axvline(1e3 * solution[key], color="black", linewidth=2.2)
    for key in [
        "z_lens1_bfl_surface",
        "z_lens1_outer_surface",
        "z_lens2_cell_side_bfl_surface",
        "z_lens2_outer_surface",
    ]:
        ax.axvline(1e3 * solution[key], color="black", linestyle="--", linewidth=1.1)

    for z_left_key, z_right_key, label in [
        ("z_lens1_outer_surface", "z_lens1_bfl_surface", "L1 AC254-040-C"),
        ("z_lens2_cell_side_bfl_surface", "z_lens2_outer_surface", "L2 AC254-040-C"),
    ]:
        z_left = 1e3 * solution[z_left_key]
        z_right = 1e3 * solution[z_right_key]
        y_half = 1e3 * geometry.lens.mount_half_height
        ax.add_patch(
            Rectangle(
                (z_left, -y_half),
                z_right - z_left,
                2 * y_half,
                facecolor="0.8",
                edgecolor="0.2",
                linewidth=1.2,
                alpha=0.35,
            )
        )
        ax.text(0.5 * (z_left + z_right), 0.62 * y_half, label, ha="center", va="center")

    principal_to_bfl = geometry.lens.efl - geometry.lens.bfl
    relay1_bfl_surface_mm = 1e3 * (mot_solution["z_relay1"] - principal_to_bfl)
    relay1_outer_surface_mm = relay1_bfl_surface_mm + 1e3 * geometry.lens.center_thickness
    ax.axvline(1e3 * mot_solution["z_relay1"], color="C2", linewidth=2.0, linestyle=":")
    ax.axvline(relay1_bfl_surface_mm, color="C2", linewidth=1.0, linestyle="--")
    ax.axvline(relay1_outer_surface_mm, color="C2", linewidth=1.0, linestyle="--")
    ax.add_patch(
        Rectangle(
            (relay1_bfl_surface_mm, -1e3 * geometry.lens.mount_half_height),
            relay1_outer_surface_mm - relay1_bfl_surface_mm,
            2e3 * geometry.lens.mount_half_height,
            facecolor="C2",
            edgecolor="C2",
            linewidth=1.1,
            alpha=0.15,
        )
    )
    ax.text(1e3 * mot_solution["z_relay1"], -25, "780 4F lens 1", color="C2", ha="center", va="top", fontsize=9)

    cube_left_mm = 1e3 * solution["z_lens2_outer_surface"]
    ax.add_patch(
        Rectangle(
            (cube_left_mm, -cube_half_height_mm),
            CUBE_SIZE_MM,
            2 * cube_half_height_mm,
            facecolor="0.05",
            edgecolor="black",
            linewidth=1.4,
            alpha=0.30,
        )
    )
    ax.text(
        cube_left_mm + 0.5 * CUBE_SIZE_MM,
        cube_half_height_mm + 2.0,
        f"right-angle cube\n{CUBE_SIZE_MM:.0f} mm",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.axvline(1e3 * solution["z_mirror"], color="gray", linewidth=2.2)
    ax.axvline(0.0, color="0.5", linewidth=1.0, linestyle=":")
    ax.axvline(1e3 * solution["z_focus1"], color="C0", linestyle=":", linewidth=1.7)
    ax.axvline(1e3 * solution["z_focus2"], color="C1", linestyle=":", linewidth=1.7)
    ax.axvline(1e3 * mot_solution["z_focus"], color="C2", linestyle=":", linewidth=1.7)
    ax.axvline(1e3 * mot_solution["z_relay_focus"], color="C4", linestyle=":", linewidth=1.4)
    ax.plot(1e3 * solution["z_focus1"], 0.0, marker="o", color="C0")
    ax.plot(1e3 * solution["z_focus2"], 0.0, marker="o", color="C1")
    ax.plot(1e3 * mot_solution["z_focus"], 0.0, marker="D", color="C2")
    ax.plot(1e3 * mot_solution["z_relay_focus"], 0.0, marker="x", color="C4")

    ax.text(1e3 * solution["z_lens1"], 34, "L1 principal", ha="center")
    ax.text(1e3 * solution["z_lens2"] - 2, 34, "L2 principal", ha="right")
    ax.text(1e3 * solution["z_mirror"] + 2, 29, "1530 mirror", ha="left")
    ax.text(0, 31, "z = 0\ncell center", ha="center")
    ax.text(1e3 * solution["z_focus1"] - 6, -39, f"focus 1\n{1e3 * solution['z_focus1']:.2f} mm", color="C0", ha="right", va="top")
    ax.text(1e3 * solution["z_focus2"] + 6, -39, f"focus 2\n{1e3 * solution['z_focus2']:.2f} mm", color="C1", ha="left", va="top")
    ax.text(1e3 * mot_solution["z_focus"] - 3, 38, f"780 focus\n{1e3 * mot_solution['z_focus']:.1f} mm", color="C2", ha="right", va="top")
    ax.text(1e3 * mot_solution["z_relay_focus"], -35, "780 4F\nfocus", color="C4", ha="center", va="top", fontsize=9)

    ax.axvline(straight_limit_mm, color="red", linewidth=2.0, linestyle="--")
    ax.text(
        straight_limit_mm + 2,
        41,
        "straight-section limit\nL2 + cube left of line",
        color="red",
        ha="left",
        va="top",
    )

    ax.set_xlabel("z along normal-incidence 1530 beam axis, relative to cell center [mm]")
    ax.set_ylabel("beam radius / envelope [mm]")
    ax.set_title("1530 plus 780 normal-incidence layout; 38 mm cube red-limit model", pad=16)
    ax.set_xlim(-130, 145)
    ax.set_ylim(-45, 45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    geometry: RoundTripGeometry,
    solution: dict[str, object],
    mot_solution: dict[str, object],
) -> None:
    lens = geometry.lens
    print("1530 normal-incidence AC254-040-C geometry")
    print("------------------------------------------")
    print(f"Cell OD along beam axis          = {1e3 * geometry.cell_length_beam:.3f} mm")
    print(f"Vacuum length along beam axis    = {1e3 * geometry.vacuum_length_beam:.3f} mm")
    print(f"Wall length along beam axis      = {1e3 * geometry.wall_length_plot:.3f} mm")
    print(f"1530 input beam radius           = {1e3 * geometry.config.input_radius:.3f} mm")
    print(f"1530 input beam diameter         = {2e3 * geometry.config.input_radius:.3f} mm")
    print()
    print(f"L1 principal plane               = {1e3 * solution['z_lens1']:.3f} mm")
    print(f"L1 focus-side BFL surface        = {1e3 * solution['z_lens1_bfl_surface']:.3f} mm")
    print(f"L1 outer physical surface        = {1e3 * solution['z_lens1_outer_surface']:.3f} mm")
    print(f"L1 BFL surface to focus 1        = {1e3 * (solution['z_focus1'] - solution['z_lens1_bfl_surface']):.3f} mm")
    print(f"L1 nearest-edge coil margin      = {abs(1e3 * solution['z_lens1_bfl_surface']) - COIL_EXCLUSION_MM:.3f} mm")
    print()
    print(f"L2 principal plane               = {1e3 * solution['z_lens2']:.3f} mm")
    print(f"L2 cell-side BFL surface         = {1e3 * solution['z_lens2_cell_side_bfl_surface']:.3f} mm")
    print(f"L2 outer physical surface        = {1e3 * solution['z_lens2_outer_surface']:.3f} mm")
    print(f"L2 cell-side BFL surface to focus 2 = {1e3 * (solution['z_lens2_cell_side_bfl_surface'] - solution['z_focus2']):.3f} mm")
    print(f"L2 nearest-edge coil margin      = {1e3 * solution['z_lens2_cell_side_bfl_surface'] - COIL_EXCLUSION_MM:.3f} mm")
    print()
    print(f"1530 mirror position             = {1e3 * solution['z_mirror']:.3f} mm")
    print(f"Mirror to cell right outer edge  = {1e3 * (solution['z_mirror'] - geometry.z_outer_right):.3f} mm")
    print(f"Mirror clearance after L2 body   = {1e3 * (solution['z_mirror'] - solution['z_lens2_outer_surface']):.3f} mm")
    print(f"Right-angle cube size            = {CUBE_SIZE_MM:.3f} mm")
    print(f"Extra gap after cube before mirror = {1e3 * (solution['z_mirror'] - solution['z_lens2_outer_surface']) - CUBE_SIZE_MM:.3f} mm")
    print(f"Straight-section limit           = {STRAIGHT_SECTION_LIMIT_MM:.3f} mm")
    print(f"Cube clearance before limit      = {STRAIGHT_SECTION_LIMIT_MM - (1e3 * solution['z_lens2_outer_surface'] + CUBE_SIZE_MM):.3f} mm")
    print(f"Unfolded mirror beyond limit     = {1e3 * solution['z_mirror'] - STRAIGHT_SECTION_LIMIT_MM:.3f} mm")
    print(f"Lens EFL / BFL                   = {1e3 * lens.efl:.3f} mm / {1e3 * lens.bfl:.3f} mm")
    print(f"Actual focus 1                   = {1e3 * solution['z_focus1']:.3f} mm")
    print(f"Actual focus 2                   = {1e3 * solution['z_focus2']:.3f} mm")
    print(f"Waist at focus 1                 = {1e6 * solution['w_focus1']:.3f} um")
    print(f"Waist at focus 2                 = {1e6 * solution['w_focus2']:.3f} um")
    print()
    print("780 MOT / 1:1 4F overlay")
    print("------------------------")
    print(f"780 input diameter               = {1e3 * mot_solution['mot_diameter']:.3f} mm")
    print(f"780 4F lens 1 principal plane    = {1e3 * mot_solution['z_relay1']:.3f} mm")
    print(f"780 4F lens spacing              = {1e3 * (mot_solution['z_relay1'] - solution['z_lens2']):.3f} mm")
    print(f"780 4F intermediate focus        = {1e3 * mot_solution['z_relay_focus']:.3f} mm")
    print(f"780 beam focus after L1          = {1e3 * mot_solution['z_focus']:.3f} mm")
    print(f"780 waist at L1 focus            = {1e6 * mot_solution['w_focus']:.3f} um")


def main() -> None:
    lens = ac254_040_c_lens()
    input_1530_radius = lens.focal_length * tan(12.0 * pi / 180.0)
    fine_geometry = RoundTripGeometry(
        lens,
        GeometryConfig(
            input_radius=input_1530_radius,
            cell_aoi_deg=0.0,
            cell_outer_diameter=40e-3,
            samples_per_space=800,
        ),
    )

    z_lens1 = fine_geometry.solve_lens1_position()
    # Coil-constrained best case: place the L2 cell-side physical edge at
    # z = +38 mm, the closest allowed position with the magnetic coil present.
    z_lens2 = (COIL_EXCLUSION_MM * 1e-3) + (lens.efl - lens.bfl)
    pre_solution = {"z_lens1": z_lens1, "z_lens2": z_lens2}
    add_lens_surfaces(pre_solution, lens)
    cube_end = pre_solution["z_lens2_outer_surface"] + CUBE_SIZE_MM * 1e-3
    z_mirror = solve_mirror_for_return_focus(
        fine_geometry,
        z_lens1,
        z_lens2,
        z_min=cube_end,
    )
    result = round_trip_with_positions(fine_geometry, z_lens1, z_lens2, z_mirror)
    solution = {
        **result,
        "z_lens1": z_lens1,
        "z_lens2": z_lens2,
        "z_mirror": z_mirror,
    }
    add_lens_surfaces(solution, lens)
    mot_solution = trace_780_mot_path(fine_geometry, solution)

    output_path = Path(__file__).with_name("pmot_normal_incidence_1530_780_cube_coil_constraint.png")
    draw_1530_normal_png(fine_geometry, solution, mot_solution, output_path)
    print_summary(fine_geometry, solution, mot_solution)
    print()
    print(f"Saved PNG: {output_path}")


if __name__ == "__main__":
    main()
