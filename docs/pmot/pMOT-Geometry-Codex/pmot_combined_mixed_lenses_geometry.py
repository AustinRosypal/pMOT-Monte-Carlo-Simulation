"""Combined left-injection pMOT geometry with AC254-C lenses.

The 780 beam is assumed to be collimated by L1, then focused by L2 onto the
shared cateye mirror. L2 and the mirror are moved together by solving the 780
focus for each L2 position, then choosing the L2 position that sends the 1530
return focus to +10 mm.
"""

from __future__ import annotations

from math import asin, cos, inf, pi, sin, sqrt, tan
from pathlib import Path

from pmot_round_trip_geometry import (
    GeometryConfig,
    LensSpec,
    beam_radius_from_q,
    fused_silica_index,
    propagate_q,
    q_from_radius_and_curvature,
    waist_in_window,
)

STRAIGHT_SECTION_REFERENCE_MM = -88.905 - 25.4
COIL_EXCLUSION_MM = 36.0
MOT_WAVELENGTH = 780e-9
MOT_COLLIMATED_DIAMETER = 11.0e-3
L1_WORKING_DISTANCE_WITH_MARGIN = (32.8 + 3.0) * 1e-3
L2_WORKING_DISTANCE_WITH_MARGIN = (32.8 + 3.0) * 1e-3


def ac254_045_c_lens() -> LensSpec:
    return LensSpec(
        name="AC254-045-C",
        focal_length=45.0e-3,
        efl=45.0e-3,
        bfl=36.7e-3,
        center_thickness=12.0e-3,
        mount_length=12.0e-3,
        mount_overhang=0.0,
        mount_half_height=18e-3,
    )


def ac254_040_c_lens() -> LensSpec:
    return LensSpec(
        name="AC254-040-C",
        focal_length=40.0e-3,
        efl=40.0e-3,
        bfl=32.8e-3,
        center_thickness=12.0e-3,
        mount_length=12.0e-3,
        mount_overhang=0.0,
        mount_half_height=18e-3,
    )


class CellModel:
    def __init__(self, wavelength: float, samples_per_space: int = 500):
        self.wavelength = wavelength
        self.samples_per_space = samples_per_space
        self.n_fs = fused_silica_index(wavelength)
        self.theta_i = 0.0
        self.theta_t = asin(sin(self.theta_i) / self.n_fs)
        self.cell_outer_diameter = 40e-3
        self.cell_wall_thickness = 5e-3
        self.cell_length_beam = self.cell_outer_diameter / cos(self.theta_i)
        self.vacuum_length_beam = (
            self.cell_outer_diameter - 2 * self.cell_wall_thickness
        ) / cos(self.theta_i)
        self.wall_length_plot = self.cell_wall_thickness / cos(self.theta_i)
        self.wall_b_eff = self.cell_wall_thickness / (self.n_fs * cos(self.theta_t))
        self.z_outer_left = -0.5 * self.cell_length_beam
        self.z_outer_right = 0.5 * self.cell_length_beam
        self.z_inner_left = self.z_outer_left + self.wall_length_plot
        self.z_inner_right = self.z_outer_right - self.wall_length_plot


def add_lens_surfaces(solution: dict[str, object], lens1: LensSpec, lens2: LensSpec) -> None:
    p1_to_bfl = lens1.efl - lens1.bfl
    p2_to_bfl = lens2.efl - lens2.bfl
    z_lens1 = solution["z_lens1"]
    z_lens2 = solution["z_lens2"]
    solution["z_lens1_optical_bfl_surface"] = z_lens1 + p1_to_bfl
    solution["z_lens2_optical_bfl_surface"] = z_lens2 - p2_to_bfl
    solution["z_lens1_bfl_surface"] = solution["z_focus1"] - L1_WORKING_DISTANCE_WITH_MARGIN
    solution["z_lens1_outer_surface"] = solution["z_lens1_bfl_surface"] - lens1.center_thickness
    solution["z_lens2_outer_surface"] = solution["z_mirror"] - L2_WORKING_DISTANCE_WITH_MARGIN
    solution["z_lens2_cell_side_bfl_surface"] = solution["z_lens2_outer_surface"] - lens2.center_thickness


def forward_system(cell: CellModel, lens1: LensSpec, lens2: LensSpec, z_lens1: float, z_lens2: float, z_mirror: float) -> list[dict[str, float | str]]:
    d_before_cell = cell.z_outer_left - z_lens1
    d_after_cell = z_lens2 - cell.z_outer_right
    d_to_mirror = z_mirror - z_lens2
    if min(d_before_cell, d_after_cell, d_to_mirror) < 0:
        raise ValueError("Invalid forward geometry.")
    return [
        {"type": "lens", "f": lens1.focal_length},
        {"type": "space", "d_plot": d_before_cell, "d_abcd": d_before_cell},
        {"type": "slab", "d_plot": cell.wall_length_plot, "d_abcd": cell.wall_b_eff},
        {"type": "space", "d_plot": cell.vacuum_length_beam, "d_abcd": cell.vacuum_length_beam},
        {"type": "slab", "d_plot": cell.wall_length_plot, "d_abcd": cell.wall_b_eff},
        {"type": "space", "d_plot": d_after_cell, "d_abcd": d_after_cell},
        {"type": "lens", "f": lens2.focal_length},
        {"type": "space", "d_plot": d_to_mirror, "d_abcd": d_to_mirror},
    ]


def reverse_system(cell: CellModel, lens1: LensSpec, lens2: LensSpec, z_lens1: float, z_lens2: float, z_mirror: float) -> list[dict[str, float | str]]:
    d_from_mirror = z_mirror - z_lens2
    d_to_cell = z_lens2 - cell.z_outer_right
    d_after_cell = cell.z_outer_left - z_lens1
    if min(d_from_mirror, d_to_cell, d_after_cell) < 0:
        raise ValueError("Invalid reverse geometry.")
    return [
        {"type": "space", "d_plot": d_from_mirror, "d_abcd": d_from_mirror},
        {"type": "lens", "f": lens2.focal_length},
        {"type": "space", "d_plot": d_to_cell, "d_abcd": d_to_cell},
        {"type": "slab", "d_plot": cell.wall_length_plot, "d_abcd": cell.wall_b_eff},
        {"type": "space", "d_plot": cell.vacuum_length_beam, "d_abcd": cell.vacuum_length_beam},
        {"type": "slab", "d_plot": cell.wall_length_plot, "d_abcd": cell.wall_b_eff},
        {"type": "space", "d_plot": d_after_cell, "d_abcd": d_after_cell},
        {"type": "lens", "f": lens1.focal_length},
        {"type": "space", "d_plot": 0.050, "d_abcd": 0.050},
    ]


def trace_1530(cell: CellModel, lens1: LensSpec, lens2: LensSpec, q_in: complex, z_lens1: float, z_lens2: float, z_mirror: float) -> dict[str, object]:
    forward = propagate_q(
        q_in,
        forward_system(cell, lens1, lens2, z_lens1, z_lens2, z_mirror),
        cell.wavelength,
        cell.samples_per_space,
    )
    reverse = propagate_q(
        forward["q_final"],
        reverse_system(cell, lens1, lens2, z_lens1, z_lens2, z_mirror),
        cell.wavelength,
        cell.samples_per_space,
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


def trace_780(cell: CellModel, lens1: LensSpec, lens2: LensSpec, z_lens1: float, z_lens2: float) -> dict[str, object]:
    q_after_l1 = q_from_radius_and_curvature(MOT_COLLIMATED_DIAMETER / 2, inf, MOT_WAVELENGTH)
    system = [
        {"type": "space", "d_plot": cell.z_outer_left - z_lens1, "d_abcd": cell.z_outer_left - z_lens1},
        {"type": "slab", "d_plot": cell.wall_length_plot, "d_abcd": cell.wall_b_eff},
        {"type": "space", "d_plot": cell.vacuum_length_beam, "d_abcd": cell.vacuum_length_beam},
        {"type": "slab", "d_plot": cell.wall_length_plot, "d_abcd": cell.wall_b_eff},
        {"type": "space", "d_plot": z_lens2 - cell.z_outer_right, "d_abcd": z_lens2 - cell.z_outer_right},
        {"type": "lens", "f": lens2.focal_length},
        {"type": "space", "d_plot": 0.080, "d_abcd": 0.080},
    ]
    result = propagate_q(q_after_l1, system, MOT_WAVELENGTH, cell.samples_per_space)
    z_global = [z_lens1 + s for s in result["s"]]
    z_focus, w_focus = waist_in_window(z_global, result["w"], z_lens2, z_lens2 + 0.080)
    return {"trace": result, "z": z_global, "z_focus": z_focus, "w_focus": w_focus}


def solve_lens1_position(cell: CellModel, lens1: LensSpec, lens2: LensSpec, q_in: complex) -> float:
    low, high = -0.150, -0.040
    best = low
    for _ in range(5):
        grid = [low + i * (high - low) / 220 for i in range(221)]
        scored = []
        for z_lens1 in grid:
            try:
                result = trace_1530(cell, lens1, lens2, q_in, z_lens1, 0.080, 0.130)
            except ValueError:
                continue
            scored.append((abs(result["z_focus1"] + 10e-3), z_lens1))
        _, best = min(scored, key=lambda item: item[0])
        step = grid[1] - grid[0]
        low, high = best - 6 * step, best + 6 * step
    return best


def solve_lens2_and_mirror(cell_1530: CellModel, cell_780: CellModel, lens1: LensSpec, lens2: LensSpec, q_in: complex, z_lens1: float) -> dict[str, object]:
    z_min = 0.038
    z_max = 0.115
    best = None
    for _ in range(5):
        grid = [z_min + i * (z_max - z_min) / 160 for i in range(161)]
        scored = []
        for z_lens2 in grid:
            mot = trace_780(cell_780, lens1, lens2, z_lens1, z_lens2)
            z_mirror = mot["z_focus"]
            try:
                rt = trace_1530(cell_1530, lens1, lens2, q_in, z_lens1, z_lens2, z_mirror)
            except ValueError:
                continue
            candidate = {
                **rt,
                "z_lens1": z_lens1,
                "z_lens2": z_lens2,
                "z_mirror": z_mirror,
                "mot": mot,
                "score": abs(rt["z_focus2"] - 10e-3),
            }
            add_lens_surfaces(candidate, lens1, lens2)
            if candidate["z_lens2_cell_side_bfl_surface"] < COIL_EXCLUSION_MM * 1e-3:
                continue
            scored.append((abs(rt["z_focus2"] - 10e-3), z_lens2, z_mirror, mot, rt))
        score, z_best, z_mirror, mot, rt = min(scored, key=lambda item: item[0])
        step = grid[1] - grid[0]
        z_min = max(0.038, z_best - 6 * step)
        z_max = z_best + 6 * step
        best = {
            **rt,
            "z_lens1": z_lens1,
            "z_lens2": z_best,
            "z_mirror": z_mirror,
            "mot": mot,
            "score": score,
        }
    if best is None:
        raise RuntimeError("No solution found.")
    add_lens_surfaces(best, lens1, lens2)
    return best


def draw_geometry(cell: CellModel, lens1: LensSpec, lens2: LensSpec, solution: dict[str, object], output_path: Path) -> None:
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

    ax.plot([1e3 * z for z in z_forward], [1e3 * w for w in w_forward], color="C0", linewidth=2.0, label="1530 forward")
    ax.plot([1e3 * z for z in z_forward], [-1e3 * w for w in w_forward], color="C0", linewidth=2.0)
    ax.plot([1e3 * z for z in z_reverse], [1e3 * w for w in w_reverse], color="C1", linewidth=2.0, linestyle="--", label="1530 return")
    ax.plot([1e3 * z for z in z_reverse], [-1e3 * w for w in w_reverse], color="C1", linewidth=2.0, linestyle="--")
    ax.plot([1e3 * z for z in mot["z"]], [1e3 * w for w in mot["trace"]["w"]], color="C2", linewidth=2.0, label="780 to cateye")
    ax.plot([1e3 * z for z in mot["z"]], [-1e3 * w for w in mot["trace"]["w"]], color="C2", linewidth=2.0)

    ax.axvspan(-COIL_EXCLUSION_MM, COIL_EXCLUSION_MM, color="red", alpha=0.07, zorder=0)
    for z_limit in [-COIL_EXCLUSION_MM, COIL_EXCLUSION_MM]:
        ax.axvline(z_limit, color="red", linewidth=1.5, linestyle=":")
    ax.axvline(STRAIGHT_SECTION_REFERENCE_MM, color="red", linewidth=2.0, linestyle="--")
    ax.text(STRAIGHT_SECTION_REFERENCE_MM + 2, 43, "optical table", color="red", ha="left", va="top", fontsize=9)
    ax.text(0, 43, "coil exclusion\nno optics for |z| < 36 mm", color="red", ha="center", va="top", fontsize=9)

    glass_color = "orange"
    y_cell_half_mm = 42
    for z0, z1 in [(cell.z_outer_left, cell.z_inner_left), (cell.z_inner_right, cell.z_outer_right)]:
        ax.add_patch(Rectangle((1e3 * z0, -y_cell_half_mm), 1e3 * (z1 - z0), 2 * y_cell_half_mm, facecolor=glass_color, edgecolor=glass_color, linewidth=1.4, alpha=0.14))
    ax.add_patch(Rectangle((1e3 * cell.z_inner_left, -y_cell_half_mm), 1e3 * (cell.z_inner_right - cell.z_inner_left), 2 * y_cell_half_mm, facecolor="none", edgecolor=glass_color, linewidth=1.2, linestyle=":", alpha=0.85))

    for key in ["z_lens1", "z_lens2"]:
        ax.axvline(1e3 * solution[key], color="black", linewidth=2.2)
    for key in ["z_lens1_bfl_surface", "z_lens1_outer_surface", "z_lens2_cell_side_bfl_surface", "z_lens2_outer_surface"]:
        ax.axvline(1e3 * solution[key], color="black", linestyle="--", linewidth=1.1)

    for z_left_key, z_right_key, label, lens in [
        ("z_lens1_outer_surface", "z_lens1_bfl_surface", f"L1 {lens1.name}", lens1),
        ("z_lens2_cell_side_bfl_surface", "z_lens2_outer_surface", f"L2 {lens2.name}", lens2),
    ]:
        z_left = 1e3 * solution[z_left_key]
        z_right = 1e3 * solution[z_right_key]
        y_half = 1e3 * lens.mount_half_height
        ax.add_patch(Rectangle((z_left, -y_half), z_right - z_left, 2 * y_half, facecolor="0.8", edgecolor="0.2", linewidth=1.2, alpha=0.35))
        ax.text(0.5 * (z_left + z_right), 0.58 * y_half, label, ha="center", va="center")

    ax.axvline(1e3 * solution["z_mirror"], color="gray", linewidth=2.4)
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
    ax.text(1e3 * solution["z_focus1"] - 4, -39, f"1530 focus 1\n{1e3 * solution['z_focus1']:.2f} mm", color="C0", ha="right", va="top")
    ax.text(1e3 * solution["z_focus2"] + 4, -39, f"1530 focus 2\n{1e3 * solution['z_focus2']:.2f} mm", color="C1", ha="left", va="top")

    ax.set_xlabel("z along normal-incidence beam axis, relative to cell center [mm]")
    ax.set_ylabel("beam radius / envelope [mm]")
    ax.set_title(
        f"Left-injection 1530/780 geometry: L1 {lens1.focal_length * 1e3:.0f} mm, "
        f"L2 {lens2.focal_length * 1e3:.0f} mm",
        pad=16,
    )
    ax.set_xlim(-130, 155)
    ax.set_ylim(-45, 45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(solution: dict[str, object], lens1: LensSpec, lens2: LensSpec) -> None:
    mot = solution["mot"]
    print(f"Left-injection geometry: L1 {lens1.name}, L2 {lens2.name}")
    print("----------------------------------------------------------------")
    print(f"L1 principal plane               = {1e3 * solution['z_lens1']:.3f} mm")
    print(f"L1 optical BFL surface estimate  = {1e3 * solution['z_lens1_optical_bfl_surface']:.3f} mm")
    print(f"L1 focus-side mount end (WD+3)   = {1e3 * solution['z_lens1_bfl_surface']:.3f} mm")
    print(f"L1 outer physical surface        = {1e3 * solution['z_lens1_outer_surface']:.3f} mm")
    print(f"L1 outer-to-optical-table gap    = {1e3 * solution['z_lens1_outer_surface'] - STRAIGHT_SECTION_REFERENCE_MM:.3f} mm")
    print(f"L1 nearest-edge coil margin      = {abs(1e3 * solution['z_lens1_bfl_surface']) - COIL_EXCLUSION_MM:.3f} mm")
    print()
    print(f"L2 principal plane               = {1e3 * solution['z_lens2']:.3f} mm")
    print(f"L2 optical BFL surface estimate  = {1e3 * solution['z_lens2_optical_bfl_surface']:.3f} mm")
    print(f"L2 cell-side physical surface    = {1e3 * solution['z_lens2_cell_side_bfl_surface']:.3f} mm")
    print(f"L2 focus-side mount end (WD+3)   = {1e3 * solution['z_lens2_outer_surface']:.3f} mm")
    print(f"L2 nearest-edge coil margin      = {1e3 * solution['z_lens2_cell_side_bfl_surface'] - COIL_EXCLUSION_MM:.3f} mm")
    print(f"L1/L2 WD+3 used                  = {1e3 * L1_WORKING_DISTANCE_WITH_MARGIN:.3f} / {1e3 * L2_WORKING_DISTANCE_WITH_MARGIN:.3f} mm")
    print()
    print(f"Shared mirror / 780 cateye       = {1e3 * solution['z_mirror']:.3f} mm")
    print(f"Mirror to L2 principal           = {1e3 * (solution['z_mirror'] - solution['z_lens2']):.3f} mm")
    print(f"1530 focus 1                     = {1e3 * solution['z_focus1']:.3f} mm")
    print(f"1530 focus 2                     = {1e3 * solution['z_focus2']:.3f} mm")
    print(f"780 cateye focus                 = {1e3 * mot['z_focus']:.3f} mm")


def main() -> None:
    lens1 = ac254_045_c_lens()
    lens2 = ac254_045_c_lens()
    cell_1530 = CellModel(1530e-9, samples_per_space=500)
    cell_780 = CellModel(MOT_WAVELENGTH, samples_per_space=500)
    q_in_1530 = q_from_radius_and_curvature(lens1.focal_length * tan(12.0 * pi / 180.0), inf, 1530e-9)

    z_lens1 = solve_lens1_position(cell_1530, lens1, lens2, q_in_1530)
    solution = solve_lens2_and_mirror(cell_1530, cell_780, lens1, lens2, q_in_1530, z_lens1)
    output_path = Path(__file__).with_name("pmot_combined_mixed_lenses_geometry_l1_045_l2_045.png")
    draw_geometry(cell_1530, lens1, lens2, solution, output_path)
    print_summary(solution, lens1, lens2)
    print()
    print(f"Saved PNG: {output_path}")


if __name__ == "__main__":
    main()
