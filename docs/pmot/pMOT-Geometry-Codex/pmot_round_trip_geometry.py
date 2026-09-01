"""Standalone pMOT round-trip geometry solver and SVG plotter.

This reproduces the main output of pMOT_PLAN.ipynb without notebook state or
third-party plotting dependencies. It solves the lens positions for two target
Gaussian waists in a tilted fused-silica cell, estimates physical/mounted lens
envelopes, and writes an SVG drawing of the layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import asin, atan, cos, degrees, inf, isfinite, pi, sin, sqrt, tan
from pathlib import Path


@dataclass(frozen=True)
class LensSpec:
    name: str
    focal_length: float
    efl: float
    bfl: float
    center_thickness: float
    mount_length: float
    mount_overhang: float
    mount_half_height: float


@dataclass(frozen=True)
class GeometryConfig:
    wavelength: float = 1530e-9
    input_radius: float = 17.5e-3
    focus_separation: float = 20e-3
    cell_outer_diameter: float = 30e-3
    cell_wall_thickness: float = 5e-3
    cell_aoi_deg: float = 45.0
    mirror_gap: float = 40e-3
    extra_left_plot: float = 50e-3
    samples_per_space: int = 250
    y_plot_half_mm: float = 60.0
    y_cell_half: float = 60e-3
    pedestal_thickness: float = 5e-3


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + i * step for i in range(count)]


def fused_silica_index(wavelength_m: float) -> float:
    lam_um = wavelength_m * 1e6
    lam2 = lam_um * lam_um
    b1, b2, b3 = 0.6961663, 0.4079426, 0.8974794
    c1, c2, c3 = 0.0684043**2, 0.1162414**2, 9.896161**2
    n2 = (
        1.0
        + b1 * lam2 / (lam2 - c1)
        + b2 * lam2 / (lam2 - c2)
        + b3 * lam2 / (lam2 - c3)
    )
    return sqrt(n2)


def apply_space(q_value: complex, distance: float) -> complex:
    return q_value + distance


def apply_lens(q_value: complex, focal_length: float) -> complex:
    return q_value / (1.0 - q_value / focal_length)


def q_from_radius_and_curvature(radius: float, curvature: float, wavelength: float) -> complex:
    inv_q = (
        (0.0 if curvature == inf else 1.0 / curvature)
        - 1j * wavelength / (pi * radius * radius)
    )
    return 1.0 / inv_q


def beam_radius_from_q(q_value: complex, wavelength: float) -> float:
    inv_q = 1.0 / q_value
    return sqrt(-wavelength / (pi * inv_q.imag))


def apply_lens_inverse(q_value: complex, focal_length: float) -> complex:
    return q_value / (1.0 + q_value / focal_length)


def propagate_q(
    q0: complex,
    system: list[dict[str, float | str]],
    wavelength: float,
    samples_per_space: int,
) -> dict[str, object]:
    s_values = [0.0]
    w_values = [beam_radius_from_q(q0, wavelength)]
    q_current = q0
    s_current = 0.0

    for element in system:
        kind = element["type"]
        if kind in {"space", "slab"}:
            d_plot = float(element["d_plot"])
            d_abcd = float(element["d_abcd"])
            for ss in linspace(0.0, d_plot, samples_per_space)[1:]:
                frac = ss / d_plot if d_plot else 0.0
                q_local = apply_space(q_current, frac * d_abcd)
                s_values.append(s_current + ss)
                w_values.append(beam_radius_from_q(q_local, wavelength))
            q_current = apply_space(q_current, d_abcd)
            s_current += d_plot
        elif kind == "lens":
            q_current = apply_lens(q_current, float(element["f"]))
        else:
            raise ValueError(f"Unknown optical element: {kind}")

    return {"s": s_values, "w": w_values, "q_final": q_current}


def waist_in_window(
    z_values: list[float], w_values: list[float], z_min: float, z_max: float
) -> tuple[float, float]:
    candidates = [
        (z, w) for z, w in zip(z_values, w_values) if z_min <= z <= z_max and isfinite(w)
    ]
    if not candidates:
        return float("nan"), float("nan")
    return min(candidates, key=lambda item: item[1])


class RoundTripGeometry:
    def __init__(self, lens: LensSpec, config: GeometryConfig):
        self.lens = lens
        self.config = config
        self.n_fs = fused_silica_index(config.wavelength)
        self.theta_i = config.cell_aoi_deg * pi / 180.0
        self.theta_t = asin(sin(self.theta_i) / self.n_fs)

        self.cell_length_beam = config.cell_outer_diameter / cos(self.theta_i)
        self.vacuum_length_beam = (
            config.cell_outer_diameter - 2 * config.cell_wall_thickness
        ) / cos(self.theta_i)
        self.wall_length_plot = config.cell_wall_thickness / cos(self.theta_i)
        self.wall_b_eff = config.cell_wall_thickness / (self.n_fs * cos(self.theta_t))

        self.z_outer_left = -0.5 * self.cell_length_beam
        self.z_outer_right = 0.5 * self.cell_length_beam
        self.z_inner_left = self.z_outer_left + self.wall_length_plot
        self.z_inner_right = self.z_outer_right - self.wall_length_plot
        self.z_focus1_target = -0.5 * config.focus_separation
        self.z_focus2_target = 0.5 * config.focus_separation
        self.q_in = q_from_radius_and_curvature(config.input_radius, inf, config.wavelength)

    def build_forward_system(
        self, z_lens1: float, z_lens2: float, z_mirror: float
    ) -> list[dict[str, float | str]]:
        d_before_cell = self.z_outer_left - z_lens1
        d_after_cell = z_lens2 - self.z_outer_right
        d_to_mirror = z_mirror - z_lens2
        if min(d_before_cell, d_after_cell, d_to_mirror) < 0:
            raise ValueError("Invalid forward geometry.")
        return [
            {"type": "lens", "f": self.lens.focal_length},
            {"type": "space", "d_plot": d_before_cell, "d_abcd": d_before_cell},
            {"type": "slab", "d_plot": self.wall_length_plot, "d_abcd": self.wall_b_eff},
            {
                "type": "space",
                "d_plot": self.vacuum_length_beam,
                "d_abcd": self.vacuum_length_beam,
            },
            {"type": "slab", "d_plot": self.wall_length_plot, "d_abcd": self.wall_b_eff},
            {"type": "space", "d_plot": d_after_cell, "d_abcd": d_after_cell},
            {"type": "lens", "f": self.lens.focal_length},
            {"type": "space", "d_plot": d_to_mirror, "d_abcd": d_to_mirror},
        ]

    def build_reverse_system(
        self, z_lens1: float, z_lens2: float, z_mirror: float
    ) -> list[dict[str, float | str]]:
        d_from_mirror = z_mirror - z_lens2
        d_to_cell = z_lens2 - self.z_outer_right
        d_after_cell = self.z_outer_left - z_lens1
        if min(d_from_mirror, d_to_cell, d_after_cell) < 0:
            raise ValueError("Invalid reverse geometry.")
        return [
            {"type": "space", "d_plot": d_from_mirror, "d_abcd": d_from_mirror},
            {"type": "lens", "f": self.lens.focal_length},
            {"type": "space", "d_plot": d_to_cell, "d_abcd": d_to_cell},
            {"type": "slab", "d_plot": self.wall_length_plot, "d_abcd": self.wall_b_eff},
            {
                "type": "space",
                "d_plot": self.vacuum_length_beam,
                "d_abcd": self.vacuum_length_beam,
            },
            {"type": "slab", "d_plot": self.wall_length_plot, "d_abcd": self.wall_b_eff},
            {"type": "space", "d_plot": d_after_cell, "d_abcd": d_after_cell},
            {"type": "lens", "f": self.lens.focal_length},
            {
                "type": "space",
                "d_plot": self.config.extra_left_plot,
                "d_abcd": self.config.extra_left_plot,
            },
        ]

    def locate_forward_focus(
        self, z_lens1: float, z_lens2_test: float = 0.14, z_mirror_test: float = 0.18
    ) -> tuple[float, float]:
        result = propagate_q(
            self.q_in,
            self.build_forward_system(z_lens1, z_lens2_test, z_mirror_test),
            self.config.wavelength,
            self.config.samples_per_space,
        )
        z_global = [z_lens1 + s for s in result["s"]]
        return waist_in_window(z_global, result["w"], -0.030, 0.030)

    def round_trip_for_lens2(self, z_lens1: float, z_lens2: float) -> tuple:
        z_mirror = z_lens2 + self.config.mirror_gap
        forward = propagate_q(
            self.q_in,
            self.build_forward_system(z_lens1, z_lens2, z_mirror),
            self.config.wavelength,
            self.config.samples_per_space,
        )
        reverse = propagate_q(
            forward["q_final"],
            self.build_reverse_system(z_lens1, z_lens2, z_mirror),
            self.config.wavelength,
            self.config.samples_per_space,
        )
        z_forward = [z_lens1 + s for s in forward["s"]]
        z_reverse = [z_mirror - s for s in reverse["s"]]
        return z_mirror, forward, reverse, z_forward, z_reverse

    def return_focus_for_lens2(self, z_lens1: float, z_lens2: float) -> tuple[float, float]:
        _, _, reverse, _, z_reverse = self.round_trip_for_lens2(z_lens1, z_lens2)
        return waist_in_window(z_reverse, reverse["w"], -0.030, 0.030)

    def solve_lens1_position(self) -> float:
        low, high = -0.25, -0.04
        best = low
        for _ in range(5):
            grid = linspace(low, high, 280)
            best = min(
                grid,
                key=lambda z_l1: abs(self.locate_forward_focus(z_l1)[0] - self.z_focus1_target),
            )
            step = grid[1] - grid[0]
            low, high = best - 5 * step, best + 5 * step
        return best

    def solve_lens2_position(self, z_lens1: float) -> float:
        low, high = 0.05, 0.20
        best = low
        for _ in range(5):
            grid = linspace(low, high, 340)
            best = min(
                grid,
                key=lambda z_l2: abs(self.return_focus_for_lens2(z_lens1, z_l2)[0] - self.z_focus2_target),
            )
            step = grid[1] - grid[0]
            low, high = best - 5 * step, best + 5 * step
        return best

    def solve(self) -> dict[str, object]:
        z_lens1 = self.solve_lens1_position()
        z_lens2 = self.solve_lens2_position(z_lens1)
        z_mirror, forward, reverse, z_forward, z_reverse = self.round_trip_for_lens2(
            z_lens1, z_lens2
        )
        z_focus1, w_focus1 = waist_in_window(z_forward, forward["w"], -0.030, 0.030)
        z_focus2, w_focus2 = waist_in_window(z_reverse, reverse["w"], -0.030, 0.030)

        principal_to_bfl = self.lens.efl - self.lens.bfl
        z_lens1_bfl = z_lens1 + principal_to_bfl
        z_lens1_other = z_lens1_bfl - self.lens.center_thickness
        z_lens2_bfl = z_lens2 - principal_to_bfl
        z_lens2_other = z_lens2_bfl + self.lens.center_thickness

        z_lens1_mount_left = z_lens1_other - self.lens.mount_overhang
        z_lens1_mount_right = z_lens1_mount_left + self.lens.mount_length
        z_lens2_mount_right = z_lens2_other + self.lens.mount_overhang
        z_lens2_mount_left = z_lens2_mount_right - self.lens.mount_length

        theta1 = atan(self.config.input_radius / abs(z_focus1 - z_lens1))
        theta2 = atan(self.config.input_radius / abs(z_lens2 - z_focus2))
        theta_avg = 0.5 * (theta1 + theta2)
        capture_diameter = tan(theta_avg) * abs(z_focus2 - z_focus1)

        return {
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
            "z_lens1_bfl": z_lens1_bfl,
            "z_lens1_other": z_lens1_other,
            "z_lens2_bfl": z_lens2_bfl,
            "z_lens2_other": z_lens2_other,
            "z_lens1_mount_left": z_lens1_mount_left,
            "z_lens1_mount_right": z_lens1_mount_right,
            "z_lens2_mount_left": z_lens2_mount_left,
            "z_lens2_mount_right": z_lens2_mount_right,
            "theta1": theta1,
            "theta2": theta2,
            "theta_avg": theta_avg,
            "na1": sin(theta1),
            "na2": sin(theta2),
            "na_avg": sin(theta_avg),
            "capture_diameter": capture_diameter,
        }

    def cell_to_plot(self, u_value: float, v_value: float) -> tuple[float, float]:
        normal_angle = pi / 2 - self.theta_i
        cell_axis_angle = normal_angle + pi / 2
        z = u_value * cos(normal_angle) + v_value * cos(cell_axis_angle)
        y = u_value * sin(normal_angle) + v_value * sin(cell_axis_angle)
        return z, y


class SvgPlot:
    def __init__(self, path: Path, xlim: tuple[float, float], ylim: tuple[float, float]):
        self.path = path
        self.width = 1600
        self.height = 760
        self.left = 105
        self.right = 35
        self.top = 75
        self.bottom = 88
        self.xlim = xlim
        self.ylim = ylim
        self.parts: list[str] = []

    def x(self, data_x: float) -> float:
        x0, x1 = self.xlim
        return self.left + (data_x - x0) / (x1 - x0) * (self.width - self.left - self.right)

    def y(self, data_y: float) -> float:
        y0, y1 = self.ylim
        return self.height - self.bottom - (data_y - y0) / (y1 - y0) * (
            self.height - self.top - self.bottom
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str,
        width: float = 1.5,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{self.x(x1):.2f}" y1="{self.y(y1):.2f}" '
            f'x2="{self.x(x2):.2f}" y2="{self.y(y2):.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        stroke: str,
        width: float = 1.5,
        dash: str | None = None,
        fill: str = "none",
        opacity: float = 1.0,
    ) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{self.x(px):.2f},{self.y(py):.2f}" for px, py in points)
        self.parts.append(
            f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        stroke: str = "none",
        width: float = 1.0,
        fill: str = "none",
        opacity: float = 1.0,
    ) -> None:
        p = " ".join(f"{self.x(px):.2f},{self.y(py):.2f}" for px, py in points)
        self.parts.append(
            f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}" opacity="{opacity}"/>'
        )

    def rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        stroke: str,
        fill: str,
        width: float = 1.5,
        opacity: float = 1.0,
    ) -> None:
        sx0, sx1 = self.x(x0), self.x(x1)
        sy0, sy1 = self.y(y0), self.y(y1)
        self.parts.append(
            f'<rect x="{min(sx0, sx1):.2f}" y="{min(sy0, sy1):.2f}" '
            f'width="{abs(sx1-sx0):.2f}" height="{abs(sy1-sy0):.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" fill="{fill}" opacity="{opacity}"/>'
        )

    def text(
        self,
        x_data: float,
        y_data: float,
        value: str,
        size: int = 18,
        fill: str = "black",
        anchor: str = "middle",
        baseline: str = "middle",
    ) -> None:
        lines = value.split("\n")
        line_height = size * 1.15
        total = line_height * (len(lines) - 1)
        y_start = self.y(y_data) - total / 2
        self.parts.append(
            f'<text x="{self.x(x_data):.2f}" y="{y_start:.2f}" font-size="{size}" '
            f'font-family="Arial, Helvetica, sans-serif" fill="{fill}" '
            f'text-anchor="{anchor}" dominant-baseline="{baseline}">'
        )
        for idx, line_text in enumerate(lines):
            dy = 0 if idx == 0 else line_height
            self.parts.append(f'<tspan x="{self.x(x_data):.2f}" dy="{dy:.2f}">{escape(line_text)}</tspan>')
        self.parts.append("</text>")

    def circle(self, x_data: float, y_data: float, radius: float, fill: str) -> None:
        self.parts.append(
            f'<circle cx="{self.x(x_data):.2f}" cy="{self.y(y_data):.2f}" '
            f'r="{radius}" fill="{fill}"/>'
        )

    def draw_axes(self) -> None:
        self.parts.append('<rect width="100%" height="100%" fill="white"/>')
        self.parts.append(
            f'<rect x="{self.left}" y="{self.top}" '
            f'width="{self.width-self.left-self.right}" '
            f'height="{self.height-self.top-self.bottom}" '
            f'fill="none" stroke="black" stroke-width="1.3"/>'
        )
        for xtick in [-150, -100, -50, 0, 50, 100]:
            self.line(xtick, self.ylim[0], xtick, self.ylim[1], "#c7c7c7", 1.0, opacity=0.55)
            self.text(xtick, self.ylim[0] - 7.0, str(xtick), size=18)
        for ytick in [-60, -40, -20, 0, 20, 40, 60]:
            self.line(self.xlim[0], ytick, self.xlim[1], ytick, "#c7c7c7", 1.0, opacity=0.55)
            self.text(self.xlim[0] - 9.0, ytick, str(ytick), size=18, anchor="end")

    def write(self) -> None:
        content = "\n".join(self.parts)
        self.path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">\n'
            f"{content}\n</svg>\n"
        )


def cell_rect_points(
    geometry: RoundTripGeometry, u0: float, u1: float, v0: float, v1: float
) -> list[tuple[float, float]]:
    return [
        tuple(1000 * x for x in geometry.cell_to_plot(u, v))
        for u, v in [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    ]


def cell_line_points(
    geometry: RoundTripGeometry, u: float, v0: float, v1: float
) -> list[tuple[float, float]]:
    return [tuple(1000 * x for x in geometry.cell_to_plot(u, v)) for v in [v0, v1]]


def propagate_from_cell_center(
    q_center: complex,
    wavelength: float,
    elements: list[dict[str, float | str]],
    samples_per_space: int,
    direction: int,
) -> tuple[list[float], list[float], complex]:
    z_values = [0.0]
    w_values = [beam_radius_from_q(q_center, wavelength)]
    q_current = q_center
    z_current = 0.0

    for element in elements:
        kind = element["type"]
        if kind in {"space", "slab"}:
            d_plot = float(element["d_plot"])
            d_abcd = float(element["d_abcd"])
            sign = float(element.get("abcd_sign", 1.0))
            for ss in linspace(0.0, d_plot, samples_per_space)[1:]:
                frac = ss / d_plot if d_plot else 0.0
                q_local = apply_space(q_current, sign * frac * d_abcd)
                z_values.append(z_current + direction * ss)
                w_values.append(beam_radius_from_q(q_local, wavelength))
            q_current = apply_space(q_current, sign * d_abcd)
            z_current += direction * d_plot
        elif kind == "lens":
            if element.get("inverse", False):
                q_current = apply_lens_inverse(q_current, float(element["f"]))
            else:
                q_current = apply_lens(q_current, float(element["f"]))
        else:
            raise ValueError(f"Unknown 780 element: {kind}")

    return z_values, w_values, q_current


def solve_780_cat_eye(
    geometry: RoundTripGeometry,
    solution: dict[str, object],
    mot_beam_diameter: float = 12.7e-3,
    wavelength: float = 780e-9,
) -> dict[str, object]:
    """Solve the overlaid 780 nm cat-eye path with fixed 1530 geometry."""
    cell_radius = mot_beam_diameter / 2
    n_780 = fused_silica_index(wavelength)
    theta_t_780 = asin(sin(geometry.theta_i) / n_780)
    wall_b_eff_780 = geometry.config.cell_wall_thickness / (n_780 * cos(theta_t_780))
    q_cell = q_from_radius_and_curvature(cell_radius, inf, wavelength)

    half_vacuum = geometry.vacuum_length_beam / 2
    z_lens1 = float(solution["z_lens1"])
    z_lens2 = float(solution["z_lens2"])
    distance_left_cell_to_lens1 = geometry.z_outer_left - z_lens1
    distance_right_cell_to_lens2 = z_lens2 - geometry.z_outer_right

    left_elements = [
        {"type": "space", "d_plot": half_vacuum, "d_abcd": half_vacuum},
        {"type": "slab", "d_plot": geometry.wall_length_plot, "d_abcd": wall_b_eff_780},
        {"type": "space", "d_plot": distance_left_cell_to_lens1, "d_abcd": distance_left_cell_to_lens1},
        {"type": "lens", "f": geometry.lens.focal_length},
        {"type": "space", "d_plot": 0.120, "d_abcd": 0.120},
    ]
    z_left, w_left, _ = propagate_from_cell_center(
        q_cell, wavelength, left_elements, geometry.config.samples_per_space, direction=-1
    )

    # Reconstruct the upstream right-to-left beam from the desired collimated q
    # at cell center. Negative ABCD distances are inverse propagation.
    right_elements = [
        {"type": "space", "d_plot": half_vacuum, "d_abcd": half_vacuum, "abcd_sign": -1.0},
        {"type": "slab", "d_plot": geometry.wall_length_plot, "d_abcd": wall_b_eff_780, "abcd_sign": -1.0},
        {"type": "space", "d_plot": distance_right_cell_to_lens2, "d_abcd": distance_right_cell_to_lens2, "abcd_sign": -1.0},
        {"type": "lens", "f": geometry.lens.focal_length, "inverse": True},
        {"type": "space", "d_plot": 0.120, "d_abcd": 0.120, "abcd_sign": -1.0},
    ]
    z_right, w_right, _ = propagate_from_cell_center(
        q_cell, wavelength, right_elements, geometry.config.samples_per_space, direction=1
    )

    left_focus_z, left_focus_w = waist_in_window(z_left, w_left, -0.260, z_lens1)
    right_focus_z, right_focus_w = waist_in_window(z_right, w_right, z_lens2, 0.260)
    cell_w_values = [
        w
        for z, w in zip(z_left + z_right, w_left + w_right)
        if geometry.z_outer_left <= z <= geometry.z_outer_right
    ]

    return {
        "wavelength": wavelength,
        "n_780": n_780,
        "cell_radius": cell_radius,
        "z_left": z_left,
        "w_left": w_left,
        "z_right": z_right,
        "w_right": w_right,
        "left_focus_z": left_focus_z,
        "left_focus_w": left_focus_w,
        "right_focus_z": right_focus_z,
        "right_focus_w": right_focus_w,
        "cell_w_min": min(cell_w_values),
        "cell_w_max": max(cell_w_values),
    }


def draw_geometry_svg(
    geometry: RoundTripGeometry, solution: dict[str, object], output_path: Path
) -> None:
    plot = SvgPlot(output_path, xlim=(-157, 135), ylim=(-60, 60))
    plot.draw_axes()

    zf = [1000 * z for z in solution["z_forward"]]
    wf = [1000 * w for w in solution["forward"]["w"]]
    zr = [1000 * z for z in solution["z_reverse"]]
    wr = [1000 * w for w in solution["reverse"]["w"]]
    plot.polyline(list(zip(zf, wf)), "#1f77b4", 3.0)
    plot.polyline(list(zip(zf, [-w for w in wf])), "#1f77b4", 3.0)
    plot.polyline(list(zip(zr, wr)), "#ff7f0e", 3.0, dash="13 7")
    plot.polyline(list(zip(zr, [-w for w in wr])), "#ff7f0e", 3.0, dash="13 7")

    cfg = geometry.config
    u_outer0 = -cfg.cell_outer_diameter / 2
    u_outer1 = cfg.cell_outer_diameter / 2
    u_inner0 = u_outer0 + cfg.cell_wall_thickness
    u_inner1 = u_outer1 - cfg.cell_wall_thickness
    y_half = cfg.y_cell_half
    glass_fill = "#f6a400"

    for rect in [
        (u_outer0, u_inner0, -y_half, y_half),
        (u_inner1, u_outer1, -y_half, y_half),
        (u_inner0, u_inner1, y_half - cfg.pedestal_thickness, y_half),
        (u_inner0, u_inner1, -y_half, -y_half + cfg.pedestal_thickness),
    ]:
        plot.polygon(cell_rect_points(geometry, *rect), fill=glass_fill, opacity=0.16)
        plot.polyline(cell_rect_points(geometry, *rect) + [cell_rect_points(geometry, *rect)[0]], "#f0a000", 2.0)

    pedestal_half_extra = 16e-3
    pedestal_depth = 16e-3
    for rect in [
        (u_outer0, u_outer1, y_half - cfg.pedestal_thickness, y_half + pedestal_half_extra),
        (u_outer0, u_outer1, -y_half - pedestal_half_extra, -y_half + cfg.pedestal_thickness),
        (u_inner1, u_outer1 + pedestal_depth, -y_half + cfg.pedestal_thickness, -y_half + 3 * cfg.pedestal_thickness),
    ]:
        pts = cell_rect_points(geometry, *rect)
        plot.polyline(pts + [pts[0]], "#f0a000", 2.0)

    for u in [u_outer0, u_outer1]:
        plot.polyline(cell_line_points(geometry, u, -y_half - 35e-3, y_half + 35e-3), "red", 2.5, dash="13 9")
    for u in [u_outer0, u_inner0, u_inner1, u_outer1]:
        plot.polyline(cell_line_points(geometry, u, -y_half - 15e-3, y_half + 15e-3), "#a23caf", 2.2, dash="2 6")

    for left_key, right_key, label in [
        ("z_lens1_mount_left", "z_lens1_mount_right", "L1 mounted lens"),
        ("z_lens2_mount_left", "z_lens2_mount_right", "L2 mounted lens"),
    ]:
        plot.rect(
            1000 * solution[left_key],
            -1000 * geometry.lens.mount_half_height,
            1000 * solution[right_key],
            1000 * geometry.lens.mount_half_height,
            stroke="#303030",
            fill="#d0d0d0",
            width=3.0,
            opacity=0.48,
        )
        plot.text(
            500 * (solution[left_key] + solution[right_key]),
            0.55 * 1000 * geometry.lens.mount_half_height,
            label,
            size=20,
        )

    for key in ["z_lens1", "z_lens2"]:
        plot.line(1000 * solution[key], -60, 1000 * solution[key], 60, "black", 4.0)
    for key in ["z_lens1_other", "z_lens1_bfl", "z_lens2_bfl", "z_lens2_other"]:
        plot.line(1000 * solution[key], -60, 1000 * solution[key], 60, "black", 2.0, dash="9 5")

    plot.line(1000 * solution["z_mirror"], -60, 1000 * solution["z_mirror"], 60, "#888888", 4.0)
    plot.line(0, -60, 0, 60, "#808080", 1.8, dash="2 4")
    plot.line(1000 * solution["z_focus1"], -60, 1000 * solution["z_focus1"], 60, "#1f77b4", 3.0, dash="2 6")
    plot.line(1000 * solution["z_focus2"], -60, 1000 * solution["z_focus2"], 60, "#ff7f0e", 3.0, dash="2 6")
    plot.circle(1000 * solution["z_focus1"], 0, 7, "#1f77b4")
    plot.circle(1000 * solution["z_focus2"], 0, 7, "#ff7f0e")

    legend_x, legend_y = -147, 54
    plot.rect(legend_x - 2, legend_y - 10, legend_x + 44, legend_y + 4, "#d7d7d7", "white", 1.5, 0.95)
    plot.line(legend_x + 5, legend_y - 2, legend_x + 18, legend_y - 2, "#1f77b4", 3.0)
    plot.text(legend_x + 21, legend_y - 2, "forward beam", size=18, anchor="start")
    plot.line(legend_x + 5, legend_y - 7, legend_x + 18, legend_y - 7, "#ff7f0e", 3.0, dash="11 6")
    plot.text(legend_x + 21, legend_y - 7, "return beam", size=18, anchor="start")

    plot.text(1000 * solution["z_lens1"], 64, f"Lens 1\n{geometry.lens.name}", size=20)
    plot.text(1000 * solution["z_lens2"], 64, "Lens 2", size=20)
    plot.text(1000 * solution["z_mirror"], 64, "Mirror\n0 deg AOI", size=20)
    plot.text(0, 48, "z = 0\ncell center", size=18)
    plot.text(1000 * solution["z_focus1"], -58, f"focus 1\n{1000 * solution['z_focus1']:.2f} mm", size=18, fill="#1f77b4")
    plot.text(1000 * solution["z_focus2"], -58, f"focus 2\n{1000 * solution['z_focus2']:.2f} mm", size=18, fill="#ff7f0e")

    plot.text(-10, 68, f"1530 nm round-trip Gaussian beam tracing; cell AOI = {geometry.config.cell_aoi_deg:.1f} deg", size=24)
    plot.text(-10, -72, "z along beam axis, relative to cell center [mm]", size=20)
    plot.text(-172, 0, "beam radius / envelope [mm]", size=20)
    plot.write()


def draw_geometry_png(
    geometry: RoundTripGeometry,
    solution: dict[str, object],
    output_path: Path,
    beam_780: dict[str, object] | None = None,
) -> None:
    """Draw the same solved layout with Matplotlib when available."""
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
        ax.annotate(
            "",
            xy=(5, 11),
            xytext=(55, 11),
            arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.8},
        )
        ax.annotate(
            "",
            xy=(55, -11),
            xytext=(5, -11),
            arrowprops={"arrowstyle": "->", "color": "#9467bd", "lw": 1.8},
        )

    cfg = geometry.config
    u_outer0 = -cfg.cell_outer_diameter / 2
    u_outer1 = cfg.cell_outer_diameter / 2
    u_inner0 = u_outer0 + cfg.cell_wall_thickness
    u_inner1 = u_outer1 - cfg.cell_wall_thickness
    y_half = cfg.y_cell_half

    def xy(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
        return [p[0] for p in points], [p[1] for p in points]

    glass_rects = [
        (u_outer0, u_inner0, -y_half, y_half),
        (u_inner1, u_outer1, -y_half, y_half),
        (u_inner0, u_inner1, y_half - cfg.pedestal_thickness, y_half),
        (u_inner0, u_inner1, -y_half, -y_half + cfg.pedestal_thickness),
    ]
    for rect in glass_rects:
        pts = cell_rect_points(geometry, *rect)
        x, y = xy(pts)
        ax.fill(x, y, facecolor="orange", edgecolor="none", alpha=0.12)
        x_closed, y_closed = xy(pts + [pts[0]])
        ax.plot(x_closed, y_closed, color="orange", linewidth=1.2)

    pedestal_half_extra = 16e-3
    pedestal_depth = 16e-3
    outline_rects = [
        (u_outer0, u_outer1, y_half - cfg.pedestal_thickness, y_half + pedestal_half_extra),
        (u_outer0, u_outer1, -y_half - pedestal_half_extra, -y_half + cfg.pedestal_thickness),
        (u_inner1, u_outer1 + pedestal_depth, -y_half + cfg.pedestal_thickness, -y_half + 3 * cfg.pedestal_thickness),
    ]
    for rect in outline_rects:
        pts = cell_rect_points(geometry, *rect)
        x_closed, y_closed = xy(pts + [pts[0]])
        ax.plot(x_closed, y_closed, color="orange", linewidth=1.2)

    for u in [u_outer0, u_outer1]:
        x, y = xy(cell_line_points(geometry, u, -y_half - 35e-3, y_half + 35e-3))
        ax.plot(x, y, color="red", linestyle=(0, (8, 6)), linewidth=1.4)
    for u in [u_outer0, u_inner0, u_inner1, u_outer1]:
        x, y = xy(cell_line_points(geometry, u, -y_half - 15e-3, y_half + 15e-3))
        ax.plot(x, y, color="purple", linestyle=":", linewidth=1.3)

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
                facecolor="0.8",
                edgecolor="0.1",
                linewidth=1.6,
                alpha=0.35,
            )
        )
        ax.text(0.5 * (z_left + z_right), 0.55 * y_mount, label, ha="center", va="center")

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
        ax.text(
            1e3 * beam_780["left_focus_z"],
            -42,
            "780 cat-eye\nfilter",
            color="#2ca02c",
            ha="center",
            va="center",
        )
        ax.text(
            1e3 * beam_780["right_focus_z"],
            42,
            "780 input\nfocus",
            color="#2ca02c",
            ha="center",
            va="center",
        )

    label_y = 56 if beam_780 is not None else 60
    ax.text(1e3 * solution["z_lens1"], label_y, f"Lens 1\n{geometry.lens.name}", ha="center", va="bottom")
    ax.text(1e3 * solution["z_lens2"], label_y, "Lens 2", ha="center", va="bottom")
    ax.text(1e3 * solution["z_mirror"], label_y, "Mirror\n0 deg AOI", ha="center", va="bottom")
    ax.text(0.0, 46, "z = 0\ncell center", ha="center", va="bottom")
    ax.text(1e3 * solution["z_focus1"], -55, f"focus 1\n{1e3 * solution['z_focus1']:.2f} mm", color="C0", ha="center", va="top")
    ax.text(1e3 * solution["z_focus2"], -55, f"focus 2\n{1e3 * solution['z_focus2']:.2f} mm", color="C1", ha="center", va="top")

    ax.set_xlabel("z along beam axis, relative to cell center [mm]")
    ax.set_ylabel("beam radius / envelope [mm]")
    if beam_780 is None:
        title = f"1530 nm round-trip Gaussian beam tracing; cell AOI = {geometry.config.cell_aoi_deg:.1f} deg"
    else:
        title = (
            "1530 nm round-trip plus 780 nm cat-eye overlay; "
            f"cell AOI = {geometry.config.cell_aoi_deg:.1f} deg"
        )
    ax.set_title(title, pad=18)
    ax.set_xlim(-205 if beam_780 is not None else -157, 190 if beam_780 is not None else 135)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def print_summary(geometry: RoundTripGeometry, solution: dict[str, object]) -> None:
    print("Solved geometry")
    print("---------------")
    print(f"n_fused_silica(1530 nm)          = {geometry.n_fs:.6f}")
    print(f"cell AOI in air                  = {geometry.config.cell_aoi_deg:.3f} deg")
    print(f"cell angle inside fused silica   = {degrees(geometry.theta_t):.3f} deg")
    print()
    print(f"Lens 1 focal length              = {1000 * geometry.lens.focal_length:.3f} mm")
    print(f"Lens 2 focal length              = {1000 * geometry.lens.focal_length:.3f} mm")
    print()
    print(f"Cell OD along normal             = {1000 * geometry.config.cell_outer_diameter:.3f} mm")
    print(f"Cell OD along beam axis          = {1000 * geometry.cell_length_beam:.3f} mm")
    print(f"Vacuum length along beam axis    = {1000 * geometry.vacuum_length_beam:.3f} mm")
    print(f"Wall length along beam axis      = {1000 * geometry.wall_length_plot:.3f} mm")
    print()
    print(f"Lens 1 position                  = {1000 * solution['z_lens1']:.3f} mm")
    print(f"Lens 2 position                  = {1000 * solution['z_lens2']:.3f} mm")
    print(f"Mirror position                  = {1000 * solution['z_mirror']:.3f} mm")
    print()
    print(f"Target focus 1                   = {1000 * geometry.z_focus1_target:.3f} mm")
    print(f"Target focus 2                   = {1000 * geometry.z_focus2_target:.3f} mm")
    print(f"Actual focus 1                   = {1000 * solution['z_focus1']:.3f} mm")
    print(f"Actual focus 2                   = {1000 * solution['z_focus2']:.3f} mm")
    print()
    print(f"Waist at focus 1                 = {1e6 * solution['w_focus1']:.3f} um")
    print(f"Waist at focus 2                 = {1e6 * solution['w_focus2']:.3f} um")
    print(f"Average NA                       = {solution['na_avg']:.4f}")
    print(f"Capture diameter                 = {1000 * solution['capture_diameter']:.3f} mm")


def print_780_summary(beam_780: dict[str, object]) -> None:
    print()
    print("780 nm cat-eye overlay")
    print("----------------------")
    print(f"n_fused_silica(780 nm)           = {beam_780['n_780']:.6f}")
    print(f"Assumed MOT beam diameter        = {2000 * beam_780['cell_radius']:.3f} mm")
    print(f"Cell beam radius min/max         = {1000 * beam_780['cell_w_min']:.3f} mm / {1000 * beam_780['cell_w_max']:.3f} mm")
    print(f"Left 780 cat-eye filter position = {1000 * beam_780['left_focus_z']:.3f} mm")
    print(f"Left 780 focus waist             = {1e6 * beam_780['left_focus_w']:.3f} um")
    print(f"Right 780 input focus position   = {1000 * beam_780['right_focus_z']:.3f} mm")
    print(f"Right 780 focus waist            = {1e6 * beam_780['right_focus_w']:.3f} um")


def main() -> None:
    ac508 = LensSpec(
        name="AC508-080-C",
        focal_length=80.3e-3,
        efl=80.3e-3,
        bfl=66.9e-3,
        center_thickness=20.5e-3,
        mount_length=27.7e-3,
        mount_overhang=3.9e-3,
        mount_half_height=33e-3,
    )
    geometry = RoundTripGeometry(ac508, GeometryConfig())
    solution = geometry.solve()
    beam_780 = solve_780_cat_eye(geometry, solution, mot_beam_diameter=12.7e-3)
    output_svg = Path(__file__).with_name("pmot_round_trip_geometry_ac508.svg")
    output_png = Path(__file__).with_name("pmot_round_trip_geometry_ac508.png")
    output_780_png = Path(__file__).with_name("pmot_round_trip_geometry_ac508_with_780.png")
    draw_geometry_svg(geometry, solution, output_svg)
    draw_geometry_png(geometry, solution, output_png)
    draw_geometry_png(geometry, solution, output_780_png, beam_780=beam_780)
    print_summary(geometry, solution)
    print_780_summary(beam_780)
    print()
    print(f"Saved SVG: {output_svg}")
    print(f"Saved PNG: {output_png}")
    print(f"Saved 780 overlay PNG: {output_780_png}")


if __name__ == "__main__":
    main()
