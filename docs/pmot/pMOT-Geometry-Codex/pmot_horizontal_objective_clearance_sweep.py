"""Sweep horizontal pMOT geometries for a 12.7 mm objective-clearance beam.

The horizontal cell remains at 45 deg AOI. This sweep constrains the external
collimated 1530 nm beam diameter to 12.7 mm, then tries candidate 1 inch and
2 inch C-coated achromats using the same round-trip model as the final
horizontal solution.
"""

from __future__ import annotations

from math import atan, degrees
from pathlib import Path

from pmot_final_horizontal_solution import draw_final_horizontal_png
from pmot_round_trip_geometry import (
    GeometryConfig,
    LensSpec,
    RoundTripGeometry,
    linspace,
    solve_780_cat_eye,
)

OBJECTIVE_CLEARANCE_BEAM_DIAMETER = 12.7e-3
MIRROR_GAP = 40e-3


def candidate_lenses() -> list[LensSpec]:
    return [
        LensSpec("AC254-030-C", 30e-3, 30e-3, 22.2e-3, 9.4e-3, 15.5e-3, 3.5e-3, 18e-3),
        LensSpec("AC254-035-C", 35e-3, 35e-3, 27.4e-3, 8.7e-3, 15.5e-3, 3.5e-3, 18e-3),
        LensSpec("AC254-040-C", 40e-3, 40e-3, 32.8e-3, 7.7e-3, 15.5e-3, 3.5e-3, 18e-3),
        LensSpec("AC254-045-C", 45e-3, 45e-3, 36.7e-3, 7.7e-3, 15.5e-3, 3.5e-3, 18e-3),
        LensSpec("AC254-050-C", 50e-3, 50e-3, 41.2e-3, 7.4e-3, 15.5e-3, 3.5e-3, 18e-3),
        LensSpec("AC254-060-C", 60e-3, 60e-3, 50.5e-3, 7.2e-3, 15.5e-3, 3.5e-3, 18e-3),
        LensSpec("AC254-075-C", 75e-3, 75e-3, 64.9e-3, 7.1e-3, 15.5e-3, 3.5e-3, 18e-3),
        LensSpec("AC508-075-C", 75e-3, 75e-3, 63.0e-3, 13.1e-3, 27.7e-3, 3.9e-3, 33e-3),
        LensSpec("AC508-080-C", 80.3e-3, 80.3e-3, 66.9e-3, 12.6e-3, 27.7e-3, 3.9e-3, 33e-3),
        LensSpec("AC508-100-C", 100e-3, 100e-3, 83.0e-3, 12.8e-3, 27.7e-3, 3.9e-3, 33e-3),
        LensSpec("AC508-150-C", 150e-3, 150e-3, 118e-3, 17.7e-3, 27.7e-3, 3.9e-3, 33e-3),
        LensSpec("ACT508-200-C", 200e-3, 200e-3, 182.4e-3, 16.5e-3, 27.7e-3, 3.9e-3, 33e-3),
        LensSpec("ACT508-250-C", 250e-3, 250e-3, 235.7e-3, 12.8e-3, 27.7e-3, 3.9e-3, 33e-3),
    ]


def solve_with_wide_ranges(geometry: RoundTripGeometry) -> dict[str, object]:
    z_lens1 = solve_lens1_position_wide(geometry)
    z_lens2 = solve_lens2_position_wide(geometry, z_lens1)
    z_mirror, forward, reverse, z_forward, z_reverse = geometry.round_trip_for_lens2(
        z_lens1, z_lens2
    )
    z_focus1, w_focus1 = min(
        (
            (z, w)
            for z, w in zip(z_forward, forward["w"])
            if -0.030 <= z <= 0.030
        ),
        key=lambda item: item[1],
    )
    z_focus2, w_focus2 = min(
        (
            (z, w)
            for z, w in zip(z_reverse, reverse["w"])
            if -0.030 <= z <= 0.030
        ),
        key=lambda item: item[1],
    )

    solution = {
        "z_lens1": z_lens1,
        "z_lens2": z_lens2,
        "z_mirror": z_mirror,
        "forward": forward,
        "reverse": reverse,
        "z_forward": z_forward,
        "z_reverse": z_reverse,
        "z_focus1": z_focus1,
        "z_focus2": z_focus2,
        "w_focus1": w_focus1,
        "w_focus2": w_focus2,
    }
    add_mechanics_and_angles(geometry, solution)
    return solution


def solve_lens1_position_wide(geometry: RoundTripGeometry) -> float:
    low = -max(0.50, 2.0 * geometry.lens.focal_length)
    high = -0.025
    best = low
    for _ in range(6):
        grid = linspace(low, high, 400)
        scored = []
        for z_lens1 in grid:
            try:
                z_focus, _ = geometry.locate_forward_focus(
                    z_lens1,
                    z_lens2_test=max(0.20, 2.0 * geometry.lens.focal_length),
                    z_mirror_test=max(0.24, 2.0 * geometry.lens.focal_length + MIRROR_GAP),
                )
            except ValueError:
                continue
            scored.append((abs(z_focus - geometry.z_focus1_target), z_lens1))
        _, best = min(scored, key=lambda item: item[0])
        step = grid[1] - grid[0]
        low, high = best - 6 * step, best + 6 * step
    return best


def solve_lens2_position_wide(geometry: RoundTripGeometry, z_lens1: float) -> float:
    low = geometry.z_outer_right + 1e-3
    high = max(0.50, 2.0 * geometry.lens.focal_length)
    best = low
    for _ in range(6):
        grid = linspace(low, high, 500)
        scored = []
        for z_lens2 in grid:
            try:
                z_focus, _ = geometry.return_focus_for_lens2(z_lens1, z_lens2)
            except ValueError:
                continue
            scored.append((abs(z_focus - geometry.z_focus2_target), z_lens2))
        _, best = min(scored, key=lambda item: item[0])
        step = grid[1] - grid[0]
        low = max(geometry.z_outer_right + 1e-3, best - 6 * step)
        high = best + 6 * step
    return best


def add_mechanics_and_angles(geometry: RoundTripGeometry, solution: dict[str, object]) -> None:
    lens = geometry.lens
    principal_to_bfl = lens.efl - lens.bfl
    mount_main = lens.mount_length - lens.mount_overhang
    solution["z_lens1_bfl"] = solution["z_lens1"] + principal_to_bfl
    solution["z_lens1_other"] = solution["z_lens1_bfl"] - lens.center_thickness
    solution["z_lens2_bfl"] = solution["z_lens2"] - principal_to_bfl
    solution["z_lens2_other"] = solution["z_lens2_bfl"] + lens.center_thickness
    solution["z_lens1_mount_left"] = solution["z_lens1_bfl"] - mount_main
    solution["z_lens1_mount_right"] = solution["z_lens1_bfl"] + lens.mount_overhang
    solution["z_lens2_mount_left"] = solution["z_lens2_bfl"] - lens.mount_overhang
    solution["z_lens2_mount_right"] = solution["z_lens2_bfl"] + mount_main

    theta1 = atan(geometry.config.input_radius / abs(solution["z_focus1"] - solution["z_lens1"]))
    theta2 = atan(geometry.config.input_radius / abs(solution["z_lens2"] - solution["z_focus2"]))
    theta_avg = 0.5 * (theta1 + theta2)
    solution["theta1"] = theta1
    solution["theta2"] = theta2
    solution["theta_avg"] = theta_avg
    solution["full_angle_avg"] = 2 * theta_avg


def evaluate_lens(lens: LensSpec) -> dict[str, object]:
    geometry = RoundTripGeometry(
        lens,
        GeometryConfig(
            input_radius=OBJECTIVE_CLEARANCE_BEAM_DIAMETER / 2,
            cell_aoi_deg=45.0,
            cell_outer_diameter=30e-3,
            cell_wall_thickness=5e-3,
            focus_separation=20e-3,
            mirror_gap=MIRROR_GAP,
            samples_per_space=350,
        ),
    )
    solution = solve_with_wide_ranges(geometry)
    l1_clearance = geometry.z_outer_left - solution["z_lens1_mount_right"]
    l2_clearance = solution["z_lens2_mount_left"] - geometry.z_outer_right
    return {
        "lens": lens,
        "geometry": geometry,
        "solution": solution,
        "l1_clearance": l1_clearance,
        "l2_clearance": l2_clearance,
        "score": abs(degrees(solution["full_angle_avg"]) - 9.0),
    }


def print_table(results: list[dict[str, object]]) -> None:
    print("Horizontal objective-clearance sweep")
    print("------------------------------------")
    print(f"External collimated beam diameter = {1e3 * OBJECTIVE_CLEARANCE_BEAM_DIAMETER:.3f} mm")
    print("Cell AOI = 45 deg; target foci = +/-10 mm; mirror gap = 40 mm")
    print()
    header = (
        "lens                 f(mm)  half(deg) full(deg) "
        "L1(mm)    L2(mm)    mirror(mm)  waist1(um) waist2(um) "
        "L1 env(mm)              L2 env(mm)"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        lens = result["lens"]
        solution = result["solution"]
        print(
            f"{lens.name:<20}"
            f"{1e3 * lens.focal_length:6.1f}"
            f"{degrees(solution['theta_avg']):10.3f}"
            f"{degrees(solution['full_angle_avg']):10.3f}"
            f"{1e3 * solution['z_lens1']:9.3f}"
            f"{1e3 * solution['z_lens2']:9.3f}"
            f"{1e3 * solution['z_mirror']:11.3f}"
            f"{1e6 * solution['w_focus1']:11.3f}"
            f"{1e6 * solution['w_focus2']:11.3f}"
            f"  [{1e3 * solution['z_lens1_mount_left']:8.3f}, {1e3 * solution['z_lens1_mount_right']:8.3f}]"
            f"  [{1e3 * solution['z_lens2_mount_left']:8.3f}, {1e3 * solution['z_lens2_mount_right']:8.3f}]"
        )


def main() -> None:
    results = []
    for lens in candidate_lenses():
        try:
            results.append(evaluate_lens(lens))
        except Exception as exc:
            print(f"Skipped {lens.name}: {exc}")

    results.sort(key=lambda item: item["lens"].focal_length)
    print_table(results)

    allowed = [
        result
        for result in results
        if result["solution"]["full_angle_avg"] <= 9.5 * 3.141592653589793 / 180.0
    ]
    best = min(allowed or results, key=lambda item: item["score"])
    lens = best["lens"]
    geometry = best["geometry"]
    solution = best["solution"]
    beam_780 = solve_780_cat_eye(geometry, solution, mot_beam_diameter=12.7e-3)
    output_path = Path(__file__).with_name("HorizontalObjectiveClearance_best.png")
    draw_final_horizontal_png(geometry, solution, output_path, beam_780=beam_780)

    print()
    print("Recommended candidate")
    print("---------------------")
    print(f"Lens choice                       = {lens.name}")
    print(f"External beam diameter            = {1e3 * OBJECTIVE_CLEARANCE_BEAM_DIAMETER:.3f} mm")
    print(f"Optical access half-angle avg     = {degrees(solution['theta_avg']):.3f} deg")
    print(f"Optical access full-angle avg     = {degrees(solution['full_angle_avg']):.3f} deg")
    print(f"Lens 1 position                   = {1e3 * solution['z_lens1']:.3f} mm")
    print(f"Lens 2 position                   = {1e3 * solution['z_lens2']:.3f} mm")
    print(f"Mirror position                   = {1e3 * solution['z_mirror']:.3f} mm")
    print(f"Focus 1 / focus 2                 = {1e3 * solution['z_focus1']:.3f} / {1e3 * solution['z_focus2']:.3f} mm")
    print(f"Waist 1 / waist 2                 = {1e6 * solution['w_focus1']:.3f} / {1e6 * solution['w_focus2']:.3f} um")
    print(f"L1 mounted envelope               = {1e3 * solution['z_lens1_mount_left']:.3f} to {1e3 * solution['z_lens1_mount_right']:.3f} mm")
    print(f"L2 mounted envelope               = {1e3 * solution['z_lens2_mount_left']:.3f} to {1e3 * solution['z_lens2_mount_right']:.3f} mm")
    print(f"L1-to-cell clearance              = {1e3 * best['l1_clearance']:.3f} mm")
    print(f"L2-to-cell clearance              = {1e3 * best['l2_clearance']:.3f} mm")
    print(f"Saved plot                        = {output_path}")


if __name__ == "__main__":
    main()
