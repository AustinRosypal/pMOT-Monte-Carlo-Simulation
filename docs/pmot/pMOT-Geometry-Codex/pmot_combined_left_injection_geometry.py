"""Combined 1530/780 left-injection pMOT geometry using AC254-045-C lenses."""

from __future__ import annotations

from math import inf, pi, tan
from pathlib import Path

from pmot_round_trip_geometry import (
    GeometryConfig,
    LensSpec,
    RoundTripGeometry,
    propagate_q,
    q_from_radius_and_curvature,
    waist_in_window,
)

COIL_EXCLUSION_MM = 38.0
STRAIGHT_SECTION_LIMIT_MM = -88.905
MOT_WAVELENGTH = 780e-9
MOT_COLLIMATED_DIAMETER = 11.0e-3


def ac254_045_c_lens() -> LensSpec:
    return LensSpec(
        name="AC254-045-C",
        focal_length=45.0e-3,
        efl=45.0e-3,
        bfl=36.7e-3,
        center_thickness=15.5e-3,
        mount_length=15.5e-3,
        mount_overhang=0.0,
        mount_half_height=18e-3,
    )


def add_lens_surfaces(solution: dict[str, object], lens: LensSpec) -> None:
    principal_to_bfl = lens.efl - lens.bfl
    z_lens1 = solution["z_lens1"]
    z_lens2 = solution["z_lens2"]
    solution["z_lens1_bfl_surface"] = z_lens1 + principal_to_bfl
    solution["z_lens1_outer_surface"] = solution["z_lens1_bfl_surface"] - lens.center_thickness
    solution["z_lens2_cell_side_bfl_surface"] = z_lens2 - principal_to_bfl
    solution["z_lens2_outer_surface"] = solution["z_lens2_cell_side_bfl_surface"] + lens.center_thickness


def trace_1530_round_trip(
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


def trace_780_to_mirror_focus(
    geometry: RoundTripGeometry,
    z_lens1: float,
    z_lens2: float,
    focus_search_after_l2: float = 0.080,
) -> dict[str, object]:
    q_after_l1 = q_from_radius_and_curvature(
        MOT_COLLIMATED_DIAMETER / 2,
        inf,
        MOT_WAVELENGTH,
    )
    system = [
        {"type": "space", "d_plot": geometry.z_outer_left - z_lens1, "d_abcd": geometry.z_outer_left - z_lens1},
        {"type": "slab", "d_plot": geometry.wall_length_plot, "d_abcd": geometry.wall_b_eff},
        {"type": "space", "d_plot": geometry.vacuum_length_beam, "d_abcd": geometry.vacuum_length_beam},
        {"type": "slab", "d_plot": geometry.wall_length_plot, "d_abcd": geometry.wall_b_eff},
        {"type": "space", "d_plot": z_lens2 - geometry.z_outer_right, "d_abcd": z_lens2 - geometry.z_outer_right},
        {"type": "lens", "f": geometry.lens.focal_length},
        {"type": "space", "d_plot": focus_search_after_l2, "d_abcd": focus_search_after_l2},
    ]
    result = propagate_q(q_after_l1, system, MOT_WAVELENGTH, geometry.config.samples_per_space)
    z_global = [z_lens1 + s for s in result["s"]]
    z_focus, w_focus = waist_in_window(z_global, result["w"], z_lens2, z_lens2 + focus_search_after_l2)
    return {
        "trace": result,
        "z": z_global,
        "z_focus": z_focus,
        "w_focus": w_focus,
    }


def solve_lens2_and_cateye_mirror(
    geometry_1530: RoundTripGeometry,
    geometry_780: RoundTripGeometry,
    z_lens1: float,
) -> dict[str, object]:
    principal_to_bfl = geometry_1530.lens.efl - geometry_1530.lens.bfl
    z_lens2_min = (COIL_EXCLUSION_MM * 1e-3) + principal_to_bfl
    z_lens2_max = 0.130
    best: dict[str, object] | None = None

    for pass_index in range(4):
        points = 160 if pass_index == 0 else 120
        grid = [z_lens2_min + i * (z_lens2_max - z_lens2_min) / (points - 1) for i in range(points)]
        scored = []
        for z_lens2 in grid:
            try:
                mot = trace_780_to_mirror_focus(geometry_780, z_lens1, z_lens2)
                z_mirror = mot["z_focus"]
                rt = trace_1530_round_trip(geometry_1530, z_lens1, z_lens2, z_mirror)
            except ValueError:
                continue
            score = abs(rt["z_focus2"] - 10e-3)
            scored.append((score, z_lens2, z_mirror, mot, rt))

        if not scored:
            raise RuntimeError("No feasible L2/mirror candidates found.")

        score, z_best, z_mirror, mot, rt = min(scored, key=lambda item: item[0])
        step = grid[1] - grid[0]
        z_lens2_min = max((COIL_EXCLUSION_MM * 1e-3) + principal_to_bfl, z_best - 6 * step)
        z_lens2_max = z_best + 6 * step
        best = {
            **rt,
            "z_lens1": z_lens1,
            "z_lens2": z_best,
            "z_mirror": z_mirror,
            "mot": mot,
            "score": score,
        }

    if best is None:
        raise RuntimeError("No L2/mirror solution found.")

    add_lens_surfaces(best, geometry_1530.lens)
    return best


def draw_geometry(
    geometry: RoundTripGeometry,
    solution: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(12.5, 5.1), dpi=160)

    z_forward = solution["z_forward"]
    z_reverse = solution["z_reverse"]
    w_forward = solution["forward"]["w"]
    w_reverse = solution["reverse"]["w"]
    mot = solution["mot"]
    z_mot = mot["z"]
    w_mot = mot["trace"]["w"]

    ax.plot([1e3 * z for z in z_forward], [1e3 * w for w in w_forward], color="C0", linewidth=2.0, label="1530 forward")
    ax.plot([1e3 * z for z in z_forward], [-1e3 * w for w in w_forward], color="C0", linewidth=2.0)
    ax.plot([1e3 * z for z in z_reverse], [1e3 * w for w in w_reverse], color="C1", linewidth=2.0, linestyle="--", label="1530 return")
    ax.plot([1e3 * z for z in z_reverse], [-1e3 * w for w in w_reverse], color="C1", linewidth=2.0, linestyle="--")

    ax.plot([1e3 * z for z in z_mot], [1e3 * w for w in w_mot], color="C2", linewidth=2.0, label="780 to cateye")
    ax.plot([1e3 * z for z in z_mot], [-1e3 * w for w in w_mot], color="C2", linewidth=2.0)

    ax.axvspan(-COIL_EXCLUSION_MM, COIL_EXCLUSION_MM, color="red", alpha=0.07, zorder=0)
    for z_limit in [-COIL_EXCLUSION_MM, COIL_EXCLUSION_MM]:
        ax.axvline(z_limit, color="red", linewidth=1.5, linestyle=":")
    ax.text(0, 43, "coil exclusion\nno optics for |z| < 38 mm", color="red", ha="center", va="top", fontsize=9)

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
        ("z_lens1_outer_surface", "z_lens1_bfl_surface", "L1 AC254-045-C"),
        ("z_lens2_cell_side_bfl_surface", "z_lens2_outer_surface", "L2 AC254-045-C"),
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
        ax.text(0.5 * (z_left + z_right), 0.58 * y_half, label, ha="center", va="center")

    ax.axvline(1e3 * solution["z_mirror"], color="gray", linewidth=2.4)
    ax.axvline(STRAIGHT_SECTION_LIMIT_MM, color="red", linewidth=2.0, linestyle="--")
    ax.axvline(0.0, color="0.5", linewidth=1.0, linestyle=":")

    ax.axvline(1e3 * solution["z_focus1"], color="C0", linestyle=":", linewidth=1.7)
    ax.axvline(1e3 * solution["z_focus2"], color="C1", linestyle=":", linewidth=1.7)
    ax.axvline(1e3 * mot["z_focus"], color="C2", linestyle=":", linewidth=1.7)
    ax.plot(1e3 * solution["z_focus1"], 0.0, marker="o", color="C0")
    ax.plot(1e3 * solution["z_focus2"], 0.0, marker="o", color="C1")
    ax.plot(1e3 * mot["z_focus"], 0.0, marker="D", color="C2")

    ax.text(1e3 * solution["z_lens1"], 34, "L1 principal", ha="center")
    ax.text(1e3 * solution["z_lens2"], 34, "L2 principal", ha="center")
    ax.text(1e3 * solution["z_mirror"] + 2, 31, "shared mirror\n780 cateye", ha="left")
    ax.text(0, 31, "z = 0\ncell center", ha="center")
    ax.text(STRAIGHT_SECTION_LIMIT_MM + 2, 43, "straight-section\nreference", color="red", ha="left", va="top", fontsize=9)
    ax.text(1e3 * solution["z_focus1"] - 4, -39, f"1530 focus 1\n{1e3 * solution['z_focus1']:.2f} mm", color="C0", ha="right", va="top")
    ax.text(1e3 * solution["z_focus2"] + 4, -39, f"1530 focus 2\n{1e3 * solution['z_focus2']:.2f} mm", color="C1", ha="left", va="top")

    ax.set_xlabel("z along normal-incidence beam axis, relative to cell center [mm]")
    ax.set_ylabel("beam radius / envelope [mm]")
    ax.set_title("Combined left-injection 1530/780 geometry with AC254-045-C lenses", pad=16)
    ax.set_xlim(-90, 150)
    ax.set_ylim(-45, 45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(geometry: RoundTripGeometry, solution: dict[str, object]) -> None:
    mot = solution["mot"]
    print("Combined left-injection geometry: AC254-045-C")
    print("--------------------------------------------")
    print(f"L1 principal plane               = {1e3 * solution['z_lens1']:.3f} mm")
    print(f"L1 focus-side BFL surface        = {1e3 * solution['z_lens1_bfl_surface']:.3f} mm")
    print(f"L1 outer physical surface        = {1e3 * solution['z_lens1_outer_surface']:.3f} mm")
    print(f"L1 nearest-edge coil margin      = {abs(1e3 * solution['z_lens1_bfl_surface']) - COIL_EXCLUSION_MM:.3f} mm")
    print()
    print(f"L2 principal plane               = {1e3 * solution['z_lens2']:.3f} mm")
    print(f"L2 cell-side BFL surface         = {1e3 * solution['z_lens2_cell_side_bfl_surface']:.3f} mm")
    print(f"L2 outer physical surface        = {1e3 * solution['z_lens2_outer_surface']:.3f} mm")
    print(f"L2 nearest-edge coil margin      = {1e3 * solution['z_lens2_cell_side_bfl_surface'] - COIL_EXCLUSION_MM:.3f} mm")
    print()
    print(f"Shared mirror / 780 cateye       = {1e3 * solution['z_mirror']:.3f} mm")
    print(f"Mirror to L2 principal           = {1e3 * (solution['z_mirror'] - solution['z_lens2']):.3f} mm")
    print(f"Mirror to L2 outer surface       = {1e3 * (solution['z_mirror'] - solution['z_lens2_outer_surface']):.3f} mm")
    print(f"Mirror relative to red reference = {1e3 * solution['z_mirror'] - STRAIGHT_SECTION_LIMIT_MM:.3f} mm")
    print()
    print(f"1530 focus 1                     = {1e3 * solution['z_focus1']:.3f} mm")
    print(f"1530 focus 2                     = {1e3 * solution['z_focus2']:.3f} mm")
    print(f"1530 waist at focus 1            = {1e6 * solution['w_focus1']:.3f} um")
    print(f"1530 waist at focus 2            = {1e6 * solution['w_focus2']:.3f} um")
    print()
    print(f"780 collimated diameter after L1 = {1e3 * MOT_COLLIMATED_DIAMETER:.3f} mm")
    print(f"780 cateye focus                 = {1e3 * mot['z_focus']:.3f} mm")
    print(f"780 cateye waist                 = {1e6 * mot['w_focus']:.3f} um")


def main() -> None:
    lens = ac254_045_c_lens()
    input_1530_radius = lens.focal_length * tan(12.0 * pi / 180.0)
    geometry_1530 = RoundTripGeometry(
        lens,
        GeometryConfig(
            wavelength=1530e-9,
            input_radius=input_1530_radius,
            cell_aoi_deg=0.0,
            cell_outer_diameter=40e-3,
            samples_per_space=500,
        ),
    )
    geometry_780 = RoundTripGeometry(
        lens,
        GeometryConfig(
            wavelength=MOT_WAVELENGTH,
            input_radius=MOT_COLLIMATED_DIAMETER / 2,
            cell_aoi_deg=0.0,
            cell_outer_diameter=40e-3,
            samples_per_space=500,
        ),
    )

    z_lens1 = geometry_1530.solve_lens1_position()
    solution = solve_lens2_and_cateye_mirror(geometry_1530, geometry_780, z_lens1)

    output_path = Path(__file__).with_name("pmot_combined_left_injection_geometry_ac254_045_c_left_ref.png")
    draw_geometry(geometry_1530, solution, output_path)
    print_summary(geometry_1530, solution)
    print()
    print(f"Saved PNG: {output_path}")


if __name__ == "__main__":
    main()
